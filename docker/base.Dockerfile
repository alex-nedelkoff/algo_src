# Base image for algo_src containers
# Provides: CUDA 12.4, Python 3.11, uv, common system deps
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# System dependencies shared across all containers
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        curl \
        git \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata for dependency resolution
COPY pyproject.toml ./

# Create venv and install base deps only (source not yet available)
RUN uv venv /app/.venv --python python3.11 \
    && uv pip install --python /app/.venv/bin/python \
        numpy scipy hydra-core omegaconf

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy full source tree, then do editable install
COPY . .
RUN uv pip install --python /app/.venv/bin/python -e "."
