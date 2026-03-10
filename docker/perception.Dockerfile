# Perception / gate detection training container (GPU)
# PyTorch + torchvision for GateNet-equivalent CNN pipeline
FROM algo-src-base AS perception

# Install PyTorch (CUDA 12.4) then remaining perception deps
RUN uv pip install --python /app/.venv/bin/python \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --python /app/.venv/bin/python -e ".[perception]"

WORKDIR /app

CMD ["python", "-m", "perception"]
