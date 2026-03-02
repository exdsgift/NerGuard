#!/usr/bin/env bash
# Full NVIDIA/Nemotron-PII validation with sequential LLM routing (llama → qwen → gpt-oss).
# Models run one at a time to avoid GPU memory overload.
#
# Usage:
#   ./scripts/validate_nvidia_full.sh
#   ./scripts/validate_nvidia_full.sh --samples 500
#
# Results are saved under results/validation_<model>/ for each run.

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SAMPLES=${1:-1000}
RESULTS_ROOT="results/validation_$(date +%Y%m%d_%H%M%S)"

MODELS=(
    "llama3.1:8b"
    "qwen2.5:7b"
    "gpt-oss:20b"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

log()  { echo -e "${DIM}[$(date +'%H:%M:%S')]${RESET} $*"; }
info() { echo -e "${CYAN}[INFO]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()  { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ── Parse optional --samples argument ────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --samples=*)  SAMPLES="${arg#*=}" ;;
        --samples)    shift; SAMPLES="$1" ;;
    esac
done

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  NerGuard — Full NVIDIA Validation (${SAMPLES} samples)${RESET}"
echo -e "${BOLD}  Models: ${MODELS[*]}${RESET}"
echo -e "${BOLD}  Results root: ${RESULTS_ROOT}${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo ""

mkdir -p "${RESULTS_ROOT}"

# ── Run each model sequentially ───────────────────────────────────────────────
declare -A SUMMARIES

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL//:/_}"
    MODEL_SLUG="${MODEL_SLUG//\//_}"
    OUTPUT_DIR="${RESULTS_ROOT}/${MODEL_SLUG}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo -e "${BOLD}───────────────────────────────────────────────────────────────${RESET}"
    info "Starting validation with model: ${BOLD}${MODEL}${RESET}"
    echo -e "${BOLD}───────────────────────────────────────────────────────────────${RESET}"

    # Pull the model if not already cached
    info "Ensuring model is available in Ollama..."
    ollama pull "${MODEL}" 2>&1 | tail -1 || warn "Could not pull ${MODEL}, assuming it exists"

    # Run evaluation
    log "Launching uv run python -m src.evaluation.hybrid_evaluator ..."
    if uv run python -m src.evaluation.hybrid_evaluator \
        --ollama-model "${MODEL}" \
        --max-samples "${SAMPLES}" \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/run.log"; then
        ok "Evaluation finished for ${MODEL}"
    else
        err "Evaluation FAILED for ${MODEL}"
    fi

    # Show quick summary from saved JSON
    JSON="${OUTPUT_DIR}/metrics_summary.json"
    if [ -f "${JSON}" ]; then
        SUMMARIES["${MODEL}"]="${JSON}"
        echo ""
        info "Quick metrics for ${MODEL}:"
        python3 - <<PYEOF
import json, sys
with open("${JSON}") as f:
    d = json.load(f)
b = d["baseline"]; h = d["hybrid"]; dt = d["delta"]; r = d["routing"]
elapsed = d.get("elapsed_seconds", 0)
print(f"  Baseline  F1={b['macro_f1']:.4f}  Acc={b['accuracy']:.4f}")
print(f"  Hybrid    F1={h['macro_f1']:.4f}  Acc={h['accuracy']:.4f}  (ΔF1={dt['macro_f1']:+.4f})")
print(f"  LLM calls={r['llm_calls']}  helped={r['llm_helped']}  hurt={r['llm_hurt']}  net={r['net_corrections']:+d}")
print(f"  Elapsed:  {elapsed:.0f}s ({elapsed/60:.1f} min)")
PYEOF
    fi

    # Stop model to free GPU VRAM before next run
    info "Stopping ${MODEL} to free GPU memory..."
    ollama stop "${MODEL}" 2>/dev/null || true
    sleep 3
done

# ── Final cross-model comparison ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  FINAL CROSS-MODEL COMPARISON  (${SAMPLES} samples)${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"

python3 - <<PYEOF
import json, os, glob

results_root = "${RESULTS_ROOT}"
jsons = sorted(glob.glob(f"{results_root}/*/metrics_summary.json"))
if not jsons:
    print("  No metrics_summary.json files found.")
    exit()

header = f"{'Model':<22} {'Base F1':>8} {'Hyb F1':>8} {'ΔF1':>8} {'Net':>5} {'Time(min)':>10}"
print(header)
print("-" * len(header))
for path in jsons:
    model_name = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        d = json.load(f)
    b = d["baseline"]; h = d["hybrid"]; dt = d["delta"]; r = d["routing"]
    elapsed_min = d.get("elapsed_minutes", 0)
    net = r["net_corrections"]
    net_str = f"{net:+d}"
    print(f"  {model_name:<20} {b['macro_f1']:>8.4f} {h['macro_f1']:>8.4f} {dt['macro_f1']:>+8.4f} {net_str:>5} {elapsed_min:>9.1f}")

print()
# Save comparison table to file
comparison_path = f"{results_root}/comparison.txt"
rows = []
for path in jsons:
    model_name = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        d = json.load(f)
    rows.append({"model": model_name, **d})
with open(comparison_path, "w") as f:
    f.write(f"NVIDIA Validation Comparison — {results_root}\n")
    f.write(header + "\n" + "-" * len(header) + "\n")
    for r_d in rows:
        b = r_d["baseline"]; h = r_d["hybrid"]; dt = r_d["delta"]; rt = r_d["routing"]
        net = rt["net_corrections"]
        elapsed_min = r_d.get("elapsed_minutes", 0)
        f.write(f"  {r_d['model']:<20} {b['macro_f1']:>8.4f} {h['macro_f1']:>8.4f} {dt['macro_f1']:>+8.4f} {net:>+5d} {elapsed_min:>9.1f}\n")
print(f"Comparison saved to: {comparison_path}")
PYEOF

echo ""
ok "All done. Results in: ${RESULTS_ROOT}"
echo ""
