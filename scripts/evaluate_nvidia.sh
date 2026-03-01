#!/usr/bin/env bash
# Evaluate on NVIDIA/Nemotron-PII dataset
# Usage: ./scripts/evaluate_nvidia.sh
set -euo pipefail

uv run python -m src.evaluation.nvidia_evaluator
