#!/usr/bin/env bash
# NER PII Benchmark — compare NerGuard against SOTA baselines
#
# Usage:
#   ./run_benchmark.sh                                      # Run all systems × all datasets
#   ./run_benchmark.sh --systems nerguard-base,piiranha     # Subset of systems
#   ./run_benchmark.sh --datasets ai4privacy --samples 100  # Quick test
#   ./run_benchmark.sh --semantic-alignment alignments/default.json  # Enable Tier 2
#
# Systems: nerguard-base, nerguard-hybrid, piiranha, presidio, gliner, spacy, bert-ner
# Datasets: ai4privacy, nvidia-pii, wikineural
set -euo pipefail

TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
LOG_FILE="${OUTPUT_DIR}/run_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "NER PII Benchmark — ${TIMESTAMP}"
echo "=================================================="
echo "Log: ${LOG_FILE}"
echo ""

uv run python -m src.benchmark.runner "$@" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "=================================================="
echo "Benchmark complete. Results in: ${OUTPUT_DIR}"
echo "Log saved to: ${LOG_FILE}"
echo "=================================================="
