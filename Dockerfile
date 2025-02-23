# NVIDIA CUDA 12.1 베이스 이미지 사용 (Ubuntu 20.04 기반)
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 기본 패키지 및 Python 3.9 설치 + OpenGL 라이브러리 추가
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-distutils \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 기본 의존성 설치
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# torch와 torchvision 설치 (CUDA 12.1 지원)
RUN pip install torch==2.4.1 torchvision==0.19.1 -f https://download.pytorch.org/whl/cu121/torch_stable.html

# 애플리케이션 파일 복사
COPY . /app/

# static 및 output 디렉토리 생성
RUN mkdir -p /app/static /app/output

# 포트 설정
EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]