# Control / RL training container (GPU)
# PyTorch + Stable-Baselines3 for PPO policy training
FROM algo-src-base AS control

# Install PyTorch (CUDA 12.4) then remaining control + sim deps
RUN uv pip install --python /app/.venv/bin/python \
        torch --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --python /app/.venv/bin/python -e ".[control,sim,playground,logging,artifacts,viz,rendering]" \
    && uv pip install --python /app/.venv/bin/python opencv-python-headless

# PyTorch3D: build from source (no cp311 wheels on PyPI).
# --no-build-isolation lets pytorch3d see the already-installed torch.
RUN FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    uv pip install --no-build-isolation --python /app/.venv/bin/python \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git"

WORKDIR /app

CMD ["python", "-m", "training"]
