# FoundationStereo inference container (GPU)
# ViT-Large stereo depth estimation — lean build, no TensorRT/ONNX
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        curl \
        git \
        build-essential \
        ninja-build \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv for fast pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/FoundationStereo

# Install PyTorch 2.4.1 (CUDA 12.4)
RUN uv venv /opt/venv --python python3.11 \
    && uv pip install --python /opt/venv/bin/python \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu124

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install FoundationStereo deps (no open3d/jupyterlab to keep lean)
RUN uv pip install --python /opt/venv/bin/python \
        scikit-image omegaconf opencv-contrib-python imgaug timm \
        albumentations scipy joblib scikit-learn ruamel.yaml \
        trimesh pyyaml imageio transformations einops numpy

# xformers (CUDA 12.4 wheel) — must match torch version
RUN uv pip install --python /opt/venv/bin/python \
        xformers==0.0.28.post1 \
        --index-url https://download.pytorch.org/whl/cu124

# flash-attn (compile from source, needs ninja + cuda-devel)
RUN uv pip install --python /opt/venv/bin/python \
        flash-attn --no-build-isolation

# Copy FoundationStereo source (mounted or copied at build time)
COPY . .

ENV OPENCV_IO_ENABLE_OPENEXR=1

ENTRYPOINT ["python"]
