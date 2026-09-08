#!/usr/bin/env bash
set -euo pipefail
cd /home/SaturateLLM/graph_size_pilot
uv python install 3.12
uv venv --python 3.12 /home/SaturateLLM/.venv
uv pip install --python /home/SaturateLLM/.venv/bin/python 'vllm==0.28.0' 'wandb==0.29.0' 'datasets==5.0.1' 'matplotlib==3.11.1'
exec /home/SaturateLLM/.venv/bin/python -u orchestrate.py
