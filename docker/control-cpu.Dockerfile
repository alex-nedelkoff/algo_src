# Control / RL training container (CPU-only)
# Lightweight variant for local development and testing on machines without NVIDIA GPUs.
FROM algo-src-base AS control-cpu

# Install CPU-only PyTorch (much smaller than CUDA variant) then control + sim deps
RUN uv pip install --python /app/.venv/bin/python \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /app/.venv/bin/python -e ".[control,sim,tb,viz,dev]"

WORKDIR /app

CMD ["python", "-m", "pytest", "tests/"]
