# Control / RL training container (GPU)
# PyTorch + Stable-Baselines3 for PPO policy training
FROM algo-src-base AS control

# Install PyTorch (CUDA 12.4) then remaining control + sim deps
RUN uv pip install --python /app/.venv/bin/python \
        torch --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --python /app/.venv/bin/python -e ".[control,sim]"

WORKDIR /app

CMD ["python", "-m", "training"]
