#!/usr/bin/env bash
# Run local Ollama model benchmarks sequentially.
# All models write to the same --session-dir for a unified summary.
#
# Usage: bash run_local_benchmark.sh [--skip-smoke] [--models model1,model2,...]
#        bash run_local_benchmark.sh --samples=1000   # override sample count
#
# Prereqs: Ollama running, all models pulled, uv venv active.

set -euo pipefail
cd "$(dirname "$0")"

SKIP_SMOKE=false
MODELS="llama3.1:8b qwen2.5:7b qwen2.5:14b mistral-nemo:12b phi4:14b deepseek-r1:14b gpt-oss:20b"
SAMPLES=100

for arg in "$@"; do
  case $arg in
    --skip-smoke) SKIP_SMOKE=true ;;
    --models=*) MODELS="${arg#*=}" ;;
    --samples=*) SAMPLES="${arg#*=}" ;;
  esac
done

# All runs share the same session directory → single unified summary
SESSION_NAME="local_${SAMPLES}samples_$(date +%Y-%m-%d)"
SESSION_DIR="./experiments/${SESSION_NAME}"
LOG_DIR="${SESSION_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo " NerGuard Local Benchmark — $(date)"
echo " Models: $MODELS"
echo " Samples: $SAMPLES"
echo " Session dir: $SESSION_DIR"
echo "=================================================="

# ── Smoke test ──────────────────────────────────────────────────────────────

run_smoke_test() {
  local MODEL="$1"
  echo ""
  echo "--- SMOKE TEST: $MODEL ---"
  local safe_name
  safe_name="$(echo "$MODEL" | tr ':/' '__')"
  local log="$LOG_DIR/smoke_${safe_name}.log"

  if uv run python -m src.scripts.nvidia_minibatch_ollama \
       --samples 5 \
       --ollama-model "$MODEL" \
       > "$log" 2>&1; then
    # Check that at least 1 LLM call was accepted
    if grep -q "llm_accept" "$log" 2>/dev/null; then
      echo "  OK — $(tail -5 "$log" | grep -E 'f1|accept|reject' | head -3)"
    else
      echo "  WARNING: no llm_accept found in log — check $log"
    fi
  else
    echo "  FAILED — see $log"
    tail -20 "$log"
    return 1
  fi
}

# ── Benchmark run ───────────────────────────────────────────────────────────

run_benchmark() {
  local MODEL="$1"
  echo ""
  echo "=================================================="
  echo " BENCHMARK: $MODEL — $(date)"
  echo "=================================================="
  local safe_name
  safe_name="$(echo "$MODEL" | tr ':/' '__')"
  local log="$LOG_DIR/bench_${safe_name}.log"

  uv run python -m src.benchmark.runner \
    --systems nerguard-hybrid-v2 \
    --datasets nvidia-pii \
    --samples "$SAMPLES" \
    --llm-source ollama \
    --llm-model "$MODEL" \
    --no-batch-llm \
    --semantic-alignment alignments/default.json \
    --session-dir "$SESSION_DIR" \
    2>&1 | tee "$log"

  echo "  Done: $MODEL — $(date)"
}

# ── Main ────────────────────────────────────────────────────────────────────

if [ "$SKIP_SMOKE" = false ]; then
  echo ""
  echo "=== Smoke tests ==="
  for MODEL in $MODELS; do
    run_smoke_test "$MODEL" || {
      echo "  Smoke test FAILED for $MODEL — aborting"
      exit 1
    }
  done
  echo ""
  echo "=== All smoke tests passed ==="
fi

echo ""
echo "=== Starting benchmark runs ==="
for MODEL in $MODELS; do
  run_benchmark "$MODEL"
done

echo ""
echo "=================================================="
echo " ALL RUNS COMPLETED — $(date)"
echo " Results in ./experiments/"
echo " Logs in $LOG_DIR"
echo "=================================================="

# Print summary of experiment directories created
echo ""
echo "New experiment directories:"
ls -d ./experiments/20*/ 2>/dev/null | grep -v "2026-03-03_13-23" | sort
