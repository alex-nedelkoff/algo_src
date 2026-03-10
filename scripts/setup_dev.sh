#!/usr/bin/env bash
set -euo pipefail
uv venv && uv pip install -e ".[dev]" && pre-commit install
