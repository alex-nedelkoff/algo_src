# Jetson deployment container (ARM64)
# TensorRT + ONNX Runtime for onboard inference on Orin NX
#
# NOTE: This is a stub. Full Jetson deployment pipeline will be
# implemented once the perception model architecture is finalized.
FROM nvcr.io/nvidia/l4t-tensorrt:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        onnxruntime-gpu \
        numpy \
        opencv-python-headless

WORKDIR /app

COPY perception/ ./perception/

CMD ["python3", "-m", "perception"]
