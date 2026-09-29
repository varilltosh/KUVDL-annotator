# WBC Detector labeling website — Ubuntu 24.04 + CUDA
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    VISION_ROOT=/vision \
    PORT=5080

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential ninja-build git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VIRTUAL_ENV

# Same versions as the dev machine (CUDA 12.1 wheels run fine on the 12.6 base)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
        flask==3.0.2 "numpy<2" pillow pillow_heif tifffile imagecodecs scipy \
        opencv-python-headless ultralytics==8.3.237 timm transformers==4.57.1 \
        addict yapf pycocotools "supervision>=0.22.0"

# Bake the BERT text encoder GroundingDINO needs into the image (no download at first detect)
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"

WORKDIR /app
COPY app.py entrypoint.sh ./
COPY static ./static
RUN chmod +x entrypoint.sh && mkdir -p uploads labels

EXPOSE 5080
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "app.py"]
