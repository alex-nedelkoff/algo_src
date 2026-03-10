# Simulator container (CPU-only)
# For running sim environments, unit tests, and fast iteration
FROM algo-src-base AS sim

# Install sim-specific dependencies (CPU-only)
RUN uv pip install --python /app/.venv/bin/python \
        numpy \
        scipy \
        gymnasium

WORKDIR /app

CMD ["python", "-m", "sim"]
