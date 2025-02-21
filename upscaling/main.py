from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import torch
import cv2
import numpy as np
import os
from datetime import datetime
from models.network_swinir import SwinIR as net
from torch.cuda.amp import autocast


app = FastAPI()

# 출력 디렉토리 생성
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")  # /static 폴더 제공
app.mount("/output", StaticFiles(directory="output"), name="output")

# SwinIR-L 모델 로드
model_path = "models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth"
checkpoint = torch.load(model_path, map_location="cuda", weights_only=True)
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

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)