# algo_src Makefile
# Docker build targets, training shortcuts, and dev tools

COMPOSE := docker compose
BASE_TAG := algo-src-base

.PHONY: build-base build-sim build-control build-control-cpu build-perception \
        train train-perception full-stack \
        test test-sim lint jetson-export sync-artifacts help

# ---------- Docker image builds ----------

build-base: ## Build the base Docker image
	docker build -t $(BASE_TAG) -f docker/base.Dockerfile .

build-sim: build-base ## Build the simulator image
	docker build -t algo-src-sim -f docker/sim.Dockerfile .

build-control: build-base ## Build the control/RL training image (GPU)
	docker build -t algo-src-control -f docker/control.Dockerfile .

build-control-cpu: build-base ## Build the control image (CPU-only, fast)
	docker build -t algo-src-control-cpu -f docker/control-cpu.Dockerfile .

build-perception: build-base ## Build the perception training image
	docker build -t algo-src-perception -f docker/perception.Dockerfile .

# ---------- Training / run shortcuts ----------

train: build-control ## Start control RL training (GPU)
	$(COMPOSE) --profile train up

train-perception: build-perception ## Start perception training (GPU)
	$(COMPOSE) --profile train-perception up

full-stack: build-base ## Start all containers
	$(COMPOSE) --profile full-stack up

# ---------- Testing & linting ----------

test: build-control-cpu ## Run all tests in CPU container
	docker run --rm algo-src-control-cpu python -m pytest tests/ -v

test-sim: build-sim ## Run sim unit tests in container
	docker run --rm algo-src-sim python -m pytest tests/test_sim/ -v

lint: build-base ## Run ruff + mypy in container
	docker run --rm $(BASE_TAG) sh -c "uv pip install ruff mypy && ruff check . && mypy --ignore-missing-imports ."

# ---------- Artifact sync ----------

sync-artifacts: ## Upload training artifacts to R2 (RUN_DIR=outputs/...)
	python scripts/sync_artifacts.py $(RUN_DIR)

# ---------- Deployment ----------

jetson-export: build-perception ## Export ONNX model for Jetson deployment
	docker run --rm -v $$(pwd)/outputs:/app/outputs algo-src-perception \
		python -m perception.export --format onnx --output /app/outputs/model.onnx

# ---------- Help ----------

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
