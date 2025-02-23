import io
import os
import cv2
import torch
import tempfile
import numpy as np
import mediapipe as mp

from PIL import Image
from datetime import datetime
from torch.cuda.amp import autocast
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File
from models.network_swinir import SwinIR as net
from fastapi.responses import JSONResponse, FileResponse


app = FastAPI()

# 출력 디렉토리 생성
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")  # /static 폴더 제공
app.mount("/output", StaticFiles(directory="output"), name="output")

# SwinIR-L 모델 로드
model_path = "models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth"
checkpoint = torch.load(model_path, map_location="cuda", weights_only=True)

# MediaPipe의 Pose 솔루션 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3, 
    static_image_mode=False  # 비디오 스트림에서 다중 인원 감지 활성화
)

model = net(
    upscale=4,  # 기본값은 4, 동적으로 변경
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
    embed_dim=240,
    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
    mlp_ratio=2,
    upsampler="nearest+conv",
    resi_connection="3conv"
)
model.load_state_dict(checkpoint["params_ema"])
model.eval()
model = model.to("cuda")

# 해상도에 따른 정수 배율 결정
def determine_upscale_factor(width, height):
    target_width, target_height = 3840, 2160  # 4K 목표 해상도
    scale_w = target_width / width
    scale_h = target_height / height
    scale = min(scale_w, scale_h)  # 작은 비율 선택

    if scale <= 1:  # 4K 이상이면 업스케일링 없음
        return 1, True  # upscale_factor=1, only_denoise=True
    elif scale <= 2:  # 4K에 근접하도록 2배
        return 2, False
    else:  # 그 외는 4배
        return 4, False

# SwinIR 적용
# tile_size는 이미지를 영역으로 나눠서 처리 (시네벤치 같이 처리하는거)
def apply_swinir(model, img_lr, upscale_factor, tile_size=1024, overlap=128):
    with torch.no_grad():
        if upscale_factor != model.upscale:
            model.upscale = upscale_factor
        _, c, h, w = img_lr.shape
        output_h, output_w = h * upscale_factor, w * upscale_factor
        img_hr = torch.zeros(1, c, output_h, output_w, device="cuda")

        for i in range(0, h, tile_size - overlap):
            for j in range(0, w, tile_size - overlap):
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                tile = img_lr[:, :, i:i_end, j:j_end]
                with autocast():
                    tile_hr = model(tile)
                # 타일 위치에 결과 저장 (오버랩 처리 필요 시 가중치 적용)
                img_hr[:, :, i*upscale_factor:i_end*upscale_factor, 
                      j*upscale_factor:j_end*upscale_factor] = tile_hr
        return img_hr

@app.post("/upscaling")
async def upscale_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    width, height = image.size
    img_lr = np.array(image)
    img_lr = img_lr.transpose(2, 0, 1)
    img_lr = torch.from_numpy(img_lr).float() / 255.0
    img_lr = img_lr.unsqueeze(0).to("cuda")

    upscale_factor, only_denoise = determine_upscale_factor(width, height)
    img_hr = apply_swinir(model, img_lr, upscale_factor, tile_size=512)

    img_hr = img_hr.squeeze(0).cpu().numpy()
    img_hr = img_hr.transpose(1, 2, 0)
    img_hr = (img_hr * 255.0).clip(0, 255).astype(np.uint8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"upscaling_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR))

    torch.cuda.empty_cache()  # 메모리 정리
    del img_lr  # 불필요한 텐서 삭제

    upscaling_image_src = f"/output/{output_filename}"
    return JSONResponse(content={"upscaling_image_src": upscaling_image_src})

@app.post("/pose")
async def detect_pose(file: UploadFile = File(...)):
    # 업로드된 파일을 메모리에 로드
    content = await file.read()
    
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    # 임시 파일로 VideoCapture 생성
    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        os.unlink(temp_path)  # 임시 파일 삭제
        return JSONResponse(content={"error": "비디오 파일을 열 수 없습니다."}, status_code=400)

    # 비디오 속성 가져오기 (해상도, FPS 등)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 출력 비디오 객체 생성 (원래 해상도 유지, 성능 문제로 3840x2160 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"pose_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (3840, 2160))  # 해상도 증가

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 원래 해상도 유지 (리사이징 제거)
        # frame = cv2.resize(frame, (1920, 1080))  # 이 줄 제거

        # BGR에서 RGB로 변환 (MediaPipe는 RGB 입력 필요)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 포즈 감지
        results = pose.process(rgb_frame)

        # 모든 포즈 랜드마크 처리 (여러 명의 선수 탐지)
        if results.pose_landmarks:  # 기본적으로 단일 포즈 처리 (multi_pose_landmarks 대신)
            mp_drawing.draw_landmarks(
                frame,  # 출력 이미지
                results.pose_landmarks,  # 단일 포즈 랜드마크
                mp_pose.POSE_CONNECTIONS,  # 연결선 정의
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),  # 관절 점 색상
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # 연결선 색상
            )
        elif hasattr(results, 'multi_pose_landmarks') and results.multi_pose_landmarks:  # 다중 포즈 처리 (버전 확인)
            for landmarks in results.multi_pose_landmarks:  # 여러 사람의 포즈 처리
                mp_drawing.draw_landmarks(
                    frame,  # 출력 이미지
                    landmarks,  # 각 사람의 포즈 랜드마크
                    mp_pose.POSE_CONNECTIONS,  # 연결선 정의
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),  # 관절 점 색상
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # 연결선 색상
                )

        # 출력 비디오에 프레임 추가
        out.write(frame)

    # 자원 해제 및 임시 파일 삭제
    cap.release()
    out.release()
    os.unlink(temp_path)  # 임시 파일 삭제

    # JSON 응답 생성
    pose_detect_video = f"/output/{output_filename}"
    return JSONResponse(content={"pose_detect_video": pose_detect_video})

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)