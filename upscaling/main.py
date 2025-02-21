from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import time
import os
import torch
from realesrgan import RealESRGAN  # RealESRGANer로 변경
from gfpgan import GFPGANer

app = FastAPI()

app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Real-ESRGANer (4x upscaling)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_esrgan = RealESRGAN(device, scale=4)
real_esrgan.load_weights("models/RealESRGAN_x4plus.pth")

# Initialize GFPGAN (face restoration)
gfpgan = GFPGANer(
    model_path="models/GFPGANv1.4.pth",
    upscale=4,  # Real-ESRGANer와 동일한 스케일
    arch="clean",
    channel_multiplier=2,
    device=device
)

def process_image(image: Image.Image) -> Image.Image:
    """이미지를 업스케일링하고 얼굴을 개선하는 처리"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # PIL Image를 numpy 배열로 변환
    import numpy as np
    img_np = np.array(image)
    
    # Real-ESRGANer로 업스케일링 (노이즈와 블러 처리)
    upscaled_img = real_esrgan.enhance(img_np, outscale=4)[0]  # enhance 메서드 사용
    
    # GFPGAN으로 얼굴 개선
    _, _, restored_img = gfpgan.enhance(
        upscaled_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    
    # 다시 PIL Image로 변환
    return Image.fromarray(restored_img)

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upscaling-image")
async def upscaling_image(file: UploadFile = File(...)):
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"upscaled_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    
    try:
        image = Image.open(file.file)
        upscaled_image = process_image(image)
        upscaled_image.save(output_path)
        return {
            "upscaling_image": filename,
            "message": "이미지 업스케일링 성공"
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        file.file.close()