"""Generate LaTeX tables and publication-ready plots for thesis.

Reads experiment results from experiments/ directory and produces:
- LaTeX tables (booktabs) ready for copy-paste into thesis
- PDF plots with publication-quality formatting (CVPR style)

Usage:
    uv run python -m src.scripts.generate_thesis_tables
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication style — Helvetica, full-page thesis ──────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.5,
})

# matplotlib tab10 colors for consistent palette
C = plt.cm.tab10.colors

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE_DIR = os.path.join(ROOT, "Gabriele_Durante_TdL", "tables")
PLOT_DIR = os.path.join(ROOT, "Gabriele_Durante_TdL", "plots", "new")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def bold(val, is_best):
    s = f"{val:.4f}"
    return f"\\textbf{{{s}}}" if is_best else s


def bold3(val, is_best):
    s = f"{val:.3f}"
    return f"\\textbf{{{s}}}" if is_best else s


# ══════════════════════════════════════════════════════════════════════════════
#  LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════

def generate_main_benchmark():
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    systems = [
        ("NerGuard Hybrid V2 (GPT-4o)", "nerguard_hybrid_v2_gpt-4o_nvidia-pii"),
        ("Presidio", "presidio_nvidia-pii"),
        ("NerGuard Hybrid V1 (GPT-4o)", "nerguard_hybrid_gpt-4o_nvidia-pii"),
        ("Piiranha", "piiranha_nvidia-pii"),
        ("NerGuard Base", "nerguard_base_nvidia-pii"),
        ("spaCy (en\\_core\\_web\\_trf)", "spacy_en_core_web_trf_nvidia-pii"),
        ("dslim/bert-base-NER", "dslim_bert-base-ner_nvidia-pii"),
    ]

    rows = []
    for name, dirname in systems:
        r = load_json(os.path.join(exp_dir, dirname, "results.json"))
        tl = r.get("token_level", {})
        el = r.get("entity_level", {})
        lat = r.get("latency", {})
        rows.append({
            "name": name,
            "f1_macro": tl.get("f1_macro", 0),
            "f1_micro": tl.get("f1_micro", 0),
            "entity_f1": el.get("f1", 0),
            "precision": el.get("precision", 0),
            "recall": el.get("recall", 0),
            "latency": lat.get("median_ms", lat.get("mean_ms", 0)),
        })

    metrics = ["f1_macro", "f1_micro", "entity_f1", "precision", "recall"]
    best = {m: max(r[m] for r in rows) for m in metrics}

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Comparative evaluation on NVIDIA Nemotron-PII (1000 samples, Tier~2, 16 evaluable labels). Best values in \\textbf{bold}.}")
    lines.append("\\label{tab:main_benchmark_updated}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{System} & \\textbf{F1-macro} & \\textbf{F1-micro} & \\textbf{Entity-F1} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{Lat. (ms)} \\\\")
    lines.append("\\midrule")

    for r in rows:
        cols = [r["name"]]
        for m in metrics:
            cols.append(bold(r[m], r[m] == best[m]))
        cols.append(f"{r['latency']:.1f}")
        lines.append(" & ".join(cols) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(TABLE_DIR, "main_benchmark.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  [OK] main_benchmark.tex")


def generate_ablation_table():
    summary = load_json(os.path.join(
        ROOT, "experiments", "ablation_nvidia-pii_1000samples", "ablation_summary.json"
    ))

    variant_order = ["never_route", "always_route", "entropy_only", "confidence_only", "full_system"]
    variant_labels = {
        "never_route": "Never route",
        "always_route": "Always route",
        "entropy_only": "Entropy only",
        "confidence_only": "Confidence only",
        "full_system": "Full system",
    }

    rows = []
    for v in variant_order:
        r = next(s for s in summary if s["variant"] == v)
        rows.append({
            "name": variant_labels[v],
            "f1_macro": r["token_level"]["f1_macro"],
            "entity_f1": r["entity_level"]["f1"],
            "precision": r["entity_level"]["precision"],
            "recall": r["entity_level"]["recall"],
            "llm_calls": r["routing"].get("llm_calls", 0),
        })

    best_ef1 = max(r["entity_f1"] for r in rows)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study: routing component contributions (NVIDIA-PII, 1000 samples). Full system achieves optimal quality/cost tradeoff.}")
    lines.append("\\label{tab:ablation_updated}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variant} & \\textbf{F1-macro} & \\textbf{Entity-F1} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{LLM calls} \\\\")
    lines.append("\\midrule")

    for r in rows:
        ef1_str = bold(r["entity_f1"], r["entity_f1"] == best_ef1)
        line = f"{r['name']} & {r['f1_macro']:.4f} & {ef1_str} & {r['precision']:.4f} & {r['recall']:.4f} & {r['llm_calls']:,}"
        lines.append(line + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(TABLE_DIR, "ablation_study.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  [OK] ablation_study.tex")


def generate_cross_domain_table():
    domains = [
        {
            "name": "PII (NVIDIA)",
            "n_classes": 20,
            "base_path": "nvidia_pii_500samples_2026-03-05/nerguard_base_nvidia-pii",
            "hybrid_path": "nvidia_pii_500samples_2026-03-05/nerguard_hybrid_v2_gpt-4o_nvidia-pii",
            "sig": "$< 0.001$***",
        },
        {
            "name": "BC5CDR",
            "n_classes": 2,
            "base_path": "bc5cdr_500samples_2026-03-05/biomedical_base_bc5cdr",
            "hybrid_path": "bc5cdr_500samples_2026-03-05/biomedical_hybrid_gpt-4o_bc5cdr",
            "sig": "$0.024$*",
        },
        {
            "name": "BUSTER",
            "n_classes": 6,
            "base_path": "buster_500samples_2026-03-05/financial_base_buster",
            "hybrid_path": "buster_500samples_2026-03-05/financial_hybrid_gpt-4o_buster",
            "sig": "$0.225$ n.s.",
        },
        {
            "name": "FiNER-139",
            "n_classes": 139,
            "base_path": "financial_500samples/financial_base_finer-139",
            "hybrid_path": "financial_500samples/financial_hybrid_gpt-4o_finer-139",
            "sig": "---",
        },
    ]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-domain generalization: base encoder vs.~hybrid (LLM-augmented) pipeline across four NER domains (500 samples each). $p$-values from paired bootstrap test.}")
    lines.append("\\label{tab:cross_domain}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Domain} & \\textbf{\\#Classes} & \\textbf{Base E-F1} & \\textbf{Hybrid E-F1} & \\textbf{$\\Delta$} & \\textbf{$p$-value} \\\\")
    lines.append("\\midrule")

    for d in domains:
        exp_root = os.path.join(ROOT, "experiments")
        base = load_json(os.path.join(exp_root, d["base_path"], "results.json"))
        hybrid = load_json(os.path.join(exp_root, d["hybrid_path"], "results.json"))

        base_ef1 = base["entity_level"]["f1"]
        hybrid_ef1 = hybrid["entity_level"]["f1"]
        delta = hybrid_ef1 - base_ef1

        delta_str = f"{delta:+.4f}"
        if delta > 0.05:
            delta_str = f"\\textbf{{{delta_str}}}"

        line = f"{d['name']} & {d['n_classes']} & {base_ef1:.4f} & {hybrid_ef1:.4f} & {delta_str} & {d['sig']}"
        lines.append(line + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(TABLE_DIR, "cross_domain.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  [OK] cross_domain.tex")


def generate_significance_table():
    bootstrap_data = [
        ("PII", 0.5531, 0.6379, 0.0849, 0.067, 0.103, "$< 0.001$***"),
        ("BC5CDR", 0.6369, 0.6410, 0.0041, 0.001, 0.008, "$0.024$*"),
        ("BUSTER", 0.6811, 0.6769, -0.0043, -0.011, 0.003, "$0.225$ n.s."),
    ]
    mcnemar_data = [
        ("PII", 304, 780, 208.14, "$< 0.001$***"),
        ("BC5CDR", 11, 16, 0.59, "$0.441$ n.s."),
        ("BUSTER", 306, 321, 0.31, "$0.576$ n.s."),
    ]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical significance tests: base vs.~hybrid across three domains (500 samples, paired bootstrap $B=10{,}000$).}")
    lines.append("\\label{tab:significance}")
    lines.append("")
    lines.append("\\begin{subtable}{\\textwidth}")
    lines.append("\\centering")
    lines.append("\\caption{Paired bootstrap test (per-sample Entity-F1)}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Domain} & \\textbf{Base} & \\textbf{Hybrid} & \\textbf{$\\Delta$} & \\textbf{95\\% CI} & \\textbf{$p$-value} \\\\")
    lines.append("\\midrule")
    for name, base, hybrid, delta, ci_lo, ci_hi, pval in bootstrap_data:
        delta_str = f"{delta:+.4f}"
        if "***" in pval:
            delta_str = f"\\textbf{{{delta_str}}}"
        lines.append(f"{name} & {base:.4f} & {hybrid:.4f} & {delta_str} & [{ci_lo:+.3f}, {ci_hi:+.3f}] & {pval} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{subtable}")
    lines.append("")
    lines.append("\\vspace{0.3cm}")
    lines.append("")
    lines.append("\\begin{subtable}{\\textwidth}")
    lines.append("\\centering")
    lines.append("\\caption{McNemar's test (token-level discordant pairs)}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Domain} & \\textbf{Base$\\checkmark$ Hyb$\\times$} & \\textbf{Hyb$\\checkmark$ Base$\\times$} & \\textbf{$\\chi^2$} & \\textbf{$p$-value} \\\\")
    lines.append("\\midrule")
    for name, base_only, hybrid_only, chi2, pval in mcnemar_data:
        lines.append(f"{name} & {base_only} & {hybrid_only} & {chi2:.2f} & {pval} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{subtable}")
    lines.append("")
    lines.append("\\end{table}")

    with open(os.path.join(TABLE_DIR, "significance_tests.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  [OK] significance_tests.tex")


def generate_local_llms_table():
    exp_root = os.path.join(ROOT, "experiments")

    models = [
        ("GPT-4o (cloud)", "2026-03-03_13-23/nerguard_hybrid_v2_gpt-4o_nvidia-pii", "---"),
        ("qwen2.5:7b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_qwen2.5:7b_nvidia-pii", "7B"),
        ("gpt-oss:20b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_gpt-oss:20b_nvidia-pii", "20B"),
        ("deepseek-r1:14b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_deepseek-r1:14b_nvidia-pii", "14B"),
        ("llama3.1:8b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_llama3.1:8b_nvidia-pii", "8B"),
        ("phi4:14b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_phi4:14b_nvidia-pii", "14B"),
        ("qwen2.5:14b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_qwen2.5:14b_nvidia-pii", "14B"),
        ("mistral-nemo:12b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_mistral-nemo:12b_nvidia-pii", "12B"),
    ]

    rows = []
    for name, path, params in models:
        r = load_json(os.path.join(exp_root, path, "results.json"))
        tl = r.get("token_level", {})
        el = r.get("entity_level", {})
        lat = r.get("latency", {})
        rows.append({
            "name": name,
            "params": params,
            "f1_macro": tl.get("f1_macro", 0),
            "entity_f1": el.get("f1", 0),
            "latency": lat.get("mean_ms", 0),
        })

    best_f1 = max(r["f1_macro"] for r in rows)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{LLM router comparison: cloud vs.~local open-weight models (NVIDIA-PII, 1000 samples). All variants cluster within 0.030~F1-macro.}")
    lines.append("\\label{tab:llm_comparison_updated}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{LLM Model} & \\textbf{Params} & \\textbf{F1-macro} & \\textbf{Entity-F1} & \\textbf{Latency (ms)} & \\textbf{$\\Delta$ vs GPT-4o} \\\\")
    lines.append("\\midrule")

    gpt4o_f1 = rows[0]["f1_macro"]
    for r in rows:
        delta = r["f1_macro"] - gpt4o_f1
        delta_str = f"{delta:+.4f}" if r["name"] != "GPT-4o (cloud)" else "---"
        f1_str = bold(r["f1_macro"], r["f1_macro"] == best_f1)
        lat_str = f"{r['latency']:.0f}" if r["latency"] > 100 else f"{r['latency']:.1f}"
        lines.append(f"{r['name']} & {r['params']} & {f1_str} & {r['entity_f1']:.4f} & {lat_str} & {delta_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(TABLE_DIR, "local_llms.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  [OK] local_llms.tex")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS — Full-page width, CVPR style
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Precision-Recall scatter with iso-F1 curves ──────────────────────

def plot_precision_recall():
    """Scatter plot in Precision-Recall space with iso-F1 contour lines."""
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    systems = [
        ("NerGuard Hybrid V2", "nerguard_hybrid_v2_gpt-4o_nvidia-pii", C[0], "o", 110),
        ("Presidio", "presidio_nvidia-pii", C[1], "s", 90),
        ("NerGuard Hybrid V1", "nerguard_hybrid_gpt-4o_nvidia-pii", C[4], "D", 80),
        ("Piiranha", "piiranha_nvidia-pii", C[2], "^", 90),
        ("NerGuard Base", "nerguard_base_nvidia-pii", C[9], "p", 90),
        ("spaCy", "spacy_en_core_web_trf_nvidia-pii", C[5], "v", 80),
        ("dslim/BERT", "dslim_bert-base-ner_nvidia-pii", C[6], "X", 80),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Iso-F1 curves
    for f1_val in [0.4, 0.5, 0.6, 0.7, 0.8]:
        p_range = np.linspace(f1_val / 2 + 0.001, 1.0, 200)
        r_range = (f1_val * p_range) / (2 * p_range - f1_val)
        mask = (r_range > 0) & (r_range <= 1) & (p_range > 0.3)
        ax.plot(r_range[mask], p_range[mask], "-", color="0.85", linewidth=0.8, zorder=1)
        # Label on the curve
        idx = np.argmin(np.abs(p_range[mask] - 0.95))
        if idx < len(r_range[mask]):
            ax.text(r_range[mask][idx] + 0.005, 0.96, f"F1={f1_val:.1f}",
                    fontsize=8, color="0.6", ha="left", va="top")

    for name, dirname, color, marker, size in systems:
        r = load_json(os.path.join(exp_dir, dirname, "results.json"))
        el = r["entity_level"]
        ax.scatter(el["recall"], el["precision"], c=[color], marker=marker, s=size,
                   label=name, zorder=5, edgecolors="white", linewidths=0.8)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Entity-level Precision vs. Recall")
    ax.set_xlim(0.50, 0.85)
    ax.set_ylim(0.38, 0.72)
    ax.legend(loc="lower left", frameon=True, ncol=2)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "precision_recall_scatter.png"))
    plt.close()
    print("  [OK] precision_recall_scatter.pdf")


# ── Plot 2: Pareto frontier — F1 vs Latency for LLM routers ─────────────────

def plot_pareto_frontier():
    """Scatter with Pareto front: Entity-F1 vs median latency for all LLM routers."""
    exp_root = os.path.join(ROOT, "experiments")
    models = [
        ("GPT-4o\n(cloud)", "2026-03-03_13-23/nerguard_hybrid_v2_gpt-4o_nvidia-pii", "---", C[3], "*", 200),
        ("qwen2.5:7b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_qwen2.5:7b_nvidia-pii", "7B", C[0], "o", 100),
        ("gpt-oss:20b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_gpt-oss:20b_nvidia-pii", "20B", C[1], "s", 100),
        ("deepseek-r1:14b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_deepseek-r1:14b_nvidia-pii", "14B", C[2], "^", 100),
        ("llama3.1:8b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_llama3.1:8b_nvidia-pii", "8B", C[4], "D", 80),
        ("phi4:14b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_phi4:14b_nvidia-pii", "14B", C[5], "v", 80),
        ("qwen2.5:14b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_qwen2.5:14b_nvidia-pii", "14B", C[6], "p", 90),
        ("mistral-nemo:12b", "local_1000samples_2026-03-03/nerguard_hybrid_v2_mistral-nemo:12b_nvidia-pii", "12B", C[7], "X", 80),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    names, latencies, f1s = [], [], []
    for name, path, params, color, marker, size in models:
        r = load_json(os.path.join(exp_root, path, "results.json"))
        lat = r["latency"]["median_ms"]
        ef1 = r["entity_level"]["f1"]
        names.append(name)
        latencies.append(lat)
        f1s.append(ef1)
        ax.scatter(lat, ef1, c=[color], marker=marker, s=size, zorder=5,
                   edgecolors="white", linewidths=0.8)

    # Annotate each point
    offsets = {
        "GPT-4o\n(cloud)": (15, 0.0015),
        "qwen2.5:7b": (20, -0.0015),
        "gpt-oss:20b": (50, 0.001),
        "deepseek-r1:14b": (-400, 0.001),
        "llama3.1:8b": (20, -0.0015),
        "phi4:14b": (30, 0.001),
        "qwen2.5:14b": (30, -0.002),
        "mistral-nemo:12b": (20, 0.001),
    }
    for name, lat, f1 in zip(names, latencies, f1s):
        dx, dy = offsets.get(name, (20, 0.001))
        ax.annotate(name.replace("\n", " "), (lat, f1),
                    xytext=(lat + dx, f1 + dy), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5) if abs(dx) > 30 else None)

    # Pareto front (connect non-dominated points)
    pts = sorted(zip(latencies, f1s))
    pareto_lat, pareto_f1 = [pts[0][0]], [pts[0][1]]
    best_f1 = pts[0][1]
    for lat, f1 in pts[1:]:
        if f1 >= best_f1:
            pareto_lat.append(lat)
            pareto_f1.append(f1)
            best_f1 = f1
    ax.plot(pareto_lat, pareto_f1, "--", color="0.5", linewidth=1, alpha=0.6, zorder=2)

    # Highlight efficient region
    ax.axhspan(min(f1s) - 0.001, max(f1s) + 0.001, alpha=0.03, color=C[0], zorder=0)
    ax.annotate(f"$\\Delta$ = {max(f1s) - min(f1s):.4f}", xy=(0.97, 0.5),
                xycoords="axes fraction", ha="right", fontsize=10, color="0.4",
                fontstyle="italic")

    ax.set_xlabel("Median Latency (ms)")
    ax.set_ylabel("Entity-F1")
    ax.set_title("Pareto Frontier: Quality vs. Inference Cost")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pareto_frontier.png"))
    plt.close()
    print("  [OK] pareto_frontier.pdf")


# ── Plot 3: Boundary condition with regression ───────────────────────────────

def plot_boundary_condition():
    """Scatter + log-linear regression: routing gain vs number of entity classes."""
    data = load_json(os.path.join(ROOT, "plots", "boundary_n_classes_analysis.json"))

    domains = data["domains"]
    n_classes = data["n_classes"]
    deltas = data["deltas"]
    fit = data["linear_fit"]

    log_n = np.log(n_classes)
    deltas_arr = np.array(deltas)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    colors = [C[2] if d > 0 else C[3] for d in deltas]
    ax.scatter(log_n, deltas_arr, c=colors, s=140, zorder=5,
               edgecolors="black", linewidth=0.8)

    for d, x, y in zip(domains, log_n, deltas_arr):
        offset_x = 0.12
        offset_y = 0.006 if y > 0 else -0.012
        ax.annotate(d, (x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=11, fontweight="bold")

    x_fit = np.linspace(min(log_n) - 0.5, max(log_n) + 0.5, 100)
    y_fit = fit["slope"] * x_fit + fit["intercept"]
    ax.plot(x_fit, y_fit, "--", color="0.45", linewidth=1.5,
            label=f"$\\Delta = {fit['slope']:.3f} \\cdot \\ln(n) + {fit['intercept']:.3f}$\n$R^2 = {fit['r_squared']:.3f}$")

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.3)
    ax.axhspan(0, max(deltas) * 1.4, alpha=0.05, color=C[2])
    ax.axhspan(min(deltas) * 1.4, 0, alpha=0.05, color=C[3])
    ax.text(0.02, 0.95, "Hybrid wins", transform=ax.transAxes,
            fontsize=10, color=C[2], va="top", fontstyle="italic", alpha=0.7)
    ax.text(0.02, 0.05, "Base wins", transform=ax.transAxes,
            fontsize=10, color=C[3], va="bottom", fontstyle="italic", alpha=0.7)

    ax.set_xlabel("$\\ln$(Number of Entity Classes)")
    ax.set_ylabel("$\\Delta$ Entity-F1 (Hybrid $-$ Base)")
    ax.set_title("LLM Routing Effectiveness vs. Task Complexity")
    ax.legend(fontsize=10, loc="upper right")

    tick_positions = log_n
    tick_labels = [f"{int(n)}\n($\\ln$={np.log(n):.1f})" for n in n_classes]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "boundary_condition.png"))
    plt.close()
    print("  [OK] boundary_condition.pdf")


# ── Plot 4: Ablation — Lollipop chart with efficiency overlay ────────────────

def plot_ablation_lollipop():
    """Lollipop chart: Entity-F1 per ablation variant + LLM call efficiency."""
    summary = load_json(os.path.join(
        ROOT, "experiments", "ablation_nvidia-pii_1000samples", "ablation_summary.json"
    ))

    variant_order = ["never_route", "always_route", "entropy_only", "confidence_only", "full_system"]
    labels = ["Never route", "Always route", "Entropy only", "Confidence only", "Full system"]

    ef1_vals, llm_calls_vals = [], []
    for v in variant_order:
        r = next(s for s in summary if s["variant"] == v)
        ef1_vals.append(r["entity_level"]["f1"])
        llm_calls_vals.append(r["routing"].get("llm_calls", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4), gridspec_kw={"width_ratios": [3, 2]})

    # Left: horizontal lollipop for Entity-F1
    y_pos = np.arange(len(labels))[::-1]
    colors_lol = [C[7], C[3], C[0], C[0], C[2]]

    for i, (yp, val) in enumerate(zip(y_pos, ef1_vals)):
        ax1.hlines(yp, 0.64, val, color=colors_lol[i], linewidth=2.5, zorder=2)
        ax1.scatter(val, yp, color=colors_lol[i], s=120, zorder=5,
                    edgecolors="black", linewidths=0.6)
        ax1.text(val + 0.002, yp, f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0.638, 0.705)
    ax1.set_xlabel("Entity-F1")
    ax1.set_title("Quality per Variant")
    ax1.axvline(x=ef1_vals[0], color="0.8", linewidth=0.8, linestyle=":", zorder=1)
    ax1.text(ef1_vals[0] - 0.001, y_pos[0] + 0.4, "baseline", fontsize=8, color="0.5",
             ha="right", va="bottom")

    # Right: horizontal bar for LLM calls (log scale)
    for i, (yp, val) in enumerate(zip(y_pos, llm_calls_vals)):
        ax2.barh(yp, val, height=0.5, color=colors_lol[i], edgecolor="black",
                 linewidth=0.5, alpha=0.85)
        label_x = val + 100 if val < 8000 else val - 500
        ha = "left" if val < 8000 else "right"
        fc = "black" if val < 8000 else "white"
        ax2.text(label_x, yp, f"{val:,}", va="center", fontsize=9, ha=ha, color=fc, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("LLM Calls")
    ax2.set_title("Routing Cost")
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ablation_lollipop.png"))
    plt.close()
    print("  [OK] ablation_lollipop.pdf")


# ── Plot 5: Cross-domain paired dot plot (dumbbell chart) ────────────────────

def plot_cross_domain_dumbbell():
    """Dumbbell/dot plot: base vs hybrid Entity-F1 across domains."""
    domains_config = [
        ("FiNER-139\n(139 cls)", "financial_500samples/financial_base_finer-139",
         "financial_500samples/financial_hybrid_gpt-4o_finer-139", ""),
        ("BUSTER\n(6 cls)", "buster_500samples_2026-03-05/financial_base_buster",
         "buster_500samples_2026-03-05/financial_hybrid_gpt-4o_buster", "n.s."),
        ("PII\n(20 cls)", "nvidia_pii_500samples_2026-03-05/nerguard_base_nvidia-pii",
         "nvidia_pii_500samples_2026-03-05/nerguard_hybrid_v2_gpt-4o_nvidia-pii", "***"),
        ("BC5CDR\n(2 cls)", "bc5cdr_500samples_2026-03-05/biomedical_base_bc5cdr",
         "bc5cdr_500samples_2026-03-05/biomedical_hybrid_gpt-4o_bc5cdr", "*"),
    ]

    exp_root = os.path.join(ROOT, "experiments")

    fig, ax = plt.subplots(figsize=(7.5, 4))

    y_pos = np.arange(len(domains_config))

    for i, (label, base_path, hybrid_path, sig) in enumerate(domains_config):
        base = load_json(os.path.join(exp_root, base_path, "results.json"))
        hybrid = load_json(os.path.join(exp_root, hybrid_path, "results.json"))
        b_f1 = base["entity_level"]["f1"]
        h_f1 = hybrid["entity_level"]["f1"]
        delta = h_f1 - b_f1

        # Connecting line
        line_color = C[2] if delta > 0 else C[3]
        ax.plot([b_f1, h_f1], [y_pos[i], y_pos[i]], "-", color=line_color, linewidth=2.5, zorder=2)

        # Base dot
        ax.scatter(b_f1, y_pos[i], color=C[7], s=100, zorder=5,
                   edgecolors="black", linewidths=0.6, label="Base" if i == 0 else "")
        # Hybrid dot
        ax.scatter(h_f1, y_pos[i], color=C[0], s=100, zorder=5,
                   edgecolors="black", linewidths=0.6, label="Hybrid" if i == 0 else "")

        # Delta annotation
        ax.text(max(b_f1, h_f1) + 0.015, y_pos[i], f"$\\Delta$={delta:+.3f}",
                va="center", fontsize=10, color=line_color, fontweight="bold")

        # Significance marker
        if sig:
            ax.text(max(b_f1, h_f1) + 0.015, y_pos[i] - 0.25, sig,
                    va="top", fontsize=8, color="0.4")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d[0] for d in domains_config])
    ax.set_xlabel("Entity-F1")
    ax.set_title("Cross-Domain Generalization: Base vs. Hybrid Pipeline")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0.35, 0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cross_domain_dumbbell.png"))
    plt.close()
    print("  [OK] cross_domain_dumbbell.pdf")


# ── Plot 6: Significance forest plot ─────────────────────────────────────────

def plot_significance_forest():
    """Forest plot: bootstrap confidence intervals for Entity-F1 delta."""
    domains = ["PII", "BC5CDR", "BUSTER"]
    deltas = [0.0849, 0.0041, -0.0043]
    ci_lower = [0.067, 0.001, -0.011]
    ci_upper = [0.103, 0.008, 0.003]
    p_labels = ["$p < 0.001$***", "$p = 0.024$*", "$p = 0.225$ n.s."]

    fig, ax = plt.subplots(figsize=(7.5, 3))

    y_positions = np.arange(len(domains))[::-1]

    for i, (d, ci_lo, ci_hi) in enumerate(zip(deltas, ci_lower, ci_upper)):
        y = y_positions[i]
        color = C[2] if d > 0 else C[3]

        # CI bar
        ax.plot([ci_lo, ci_hi], [y, y], "-", color=color, linewidth=3, zorder=3,
                solid_capstyle="round")
        # Point estimate
        ax.scatter(d, y, color=color, s=120, zorder=5,
                   edgecolors="black", linewidths=0.8, marker="D")
        # p-value label
        ax.text(ci_hi + 0.006, y, p_labels[i], va="center", fontsize=10, color="0.35")

    ax.axvline(x=0, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(domains, fontsize=12)
    ax.set_xlabel("$\\Delta$ Entity-F1 (Hybrid $-$ Base)")
    ax.set_title("Statistical Significance: 95% Bootstrap Confidence Intervals")
    ax.set_xlim(-0.025, 0.16)

    # Shade regions
    ax.axvspan(0, 0.16, alpha=0.03, color=C[2], zorder=0)
    ax.axvspan(-0.025, 0, alpha=0.03, color=C[3], zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "significance_forest.png"))
    plt.close()
    print("  [OK] significance_forest.pdf")


# ── Plot 7: Entity frequency robustness (grouped dot plot) ───────────────────

def plot_entity_frequency_robustness():
    """Dot plot: F1-macro across tail/mid/head entity frequency buckets per system."""
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    systems = [
        ("Hybrid V2", "nerguard_hybrid_v2_gpt-4o_nvidia-pii", C[0], "o"),
        ("Base", "nerguard_base_nvidia-pii", C[7], "s"),
        ("Presidio", "presidio_nvidia-pii", C[1], "D"),
        ("Piiranha", "piiranha_nvidia-pii", C[2], "^"),
        ("spaCy", "spacy_en_core_web_trf_nvidia-pii", C[5], "v"),
        ("dslim/BERT", "dslim_bert-base-ner_nvidia-pii", C[6], "X"),
    ]

    buckets = ["tail", "mid", "head"]
    bucket_labels = ["Tail\n(rare entities)", "Mid\n(moderate freq.)", "Head\n(common entities)"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x_base = np.arange(len(buckets))
    n_systems = len(systems)
    spread = 0.08  # horizontal spread per system

    for j, (name, dirname, color, marker) in enumerate(systems):
        r = load_json(os.path.join(exp_dir, dirname, "results.json"))
        pef = r.get("per_entity_frequency", {})

        vals = []
        x_positions = []
        for k, bucket in enumerate(buckets):
            bdata = pef.get(bucket, {})
            f1 = bdata.get("f1_macro", None)
            if f1 is not None and bdata.get("n_entities", 0) > 0:
                vals.append(f1)
                x_positions.append(x_base[k] + (j - n_systems / 2) * spread)

        if vals:
            ax.plot(x_positions, vals, "-", color=color, linewidth=1, alpha=0.4, zorder=2)
            ax.scatter(x_positions, vals, color=color, marker=marker, s=80,
                       label=name, zorder=5, edgecolors="white", linewidths=0.6)

    ax.set_xticks(x_base)
    ax.set_xticklabels(bucket_labels, fontsize=11)
    ax.set_ylabel("F1-macro")
    ax.set_title("Performance by Entity Frequency: Tail vs. Head")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.set_ylim(-0.05, 1.0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "entity_frequency_robustness.png"))
    plt.close()
    print("  [OK] entity_frequency_robustness.pdf")


# ── Plot 8: Length robustness (slope chart) ──────────────────────────────────

def plot_length_robustness():
    """Connected dot plot: F1-macro degradation across input lengths per system."""
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    systems = [
        ("Hybrid V2", "nerguard_hybrid_v2_gpt-4o_nvidia-pii", C[0], "o", 2.0),
        ("Base", "nerguard_base_nvidia-pii", C[7], "s", 1.5),
        ("Presidio", "presidio_nvidia-pii", C[1], "D", 1.5),
        ("Piiranha", "piiranha_nvidia-pii", C[2], "^", 1.5),
        ("spaCy", "spacy_en_core_web_trf_nvidia-pii", C[5], "v", 1.2),
        ("dslim/BERT", "dslim_bert-base-ner_nvidia-pii", C[6], "X", 1.2),
    ]

    buckets = ["short", "medium", "long"]
    bucket_labels = ["Short\n($n \\leq 64$)", "Medium\n($64 < n \\leq 256$)", "Long\n($n > 256$)"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(buckets))

    for name, dirname, color, marker, lw in systems:
        r = load_json(os.path.join(exp_dir, dirname, "results.json"))
        plb = r.get("per_length_bucket", {})

        vals = [plb.get(b, {}).get("f1_macro", 0) for b in buckets]

        ax.plot(x, vals, "-", color=color, linewidth=lw, alpha=0.8, zorder=2)
        ax.scatter(x, vals, color=color, marker=marker, s=80,
                   label=name, zorder=5, edgecolors="white", linewidths=0.6)

    # Sample count annotations
    r0 = load_json(os.path.join(exp_dir, "nerguard_hybrid_v2_gpt-4o_nvidia-pii", "results.json"))
    plb = r0.get("per_length_bucket", {})
    for i, b in enumerate(buckets):
        n = plb.get(b, {}).get("n_samples", 0)
        ax.text(x[i], -0.02, f"n={n}", ha="center", fontsize=9, color="0.5",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=11)
    ax.set_ylabel("F1-macro")
    ax.set_title("Performance Degradation by Input Length")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.set_ylim(0.25, 0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "length_robustness.png"))
    plt.close()
    print("  [OK] length_robustness.pdf")


# ── Plot 9: Radar chart — multi-metric system comparison ─────────────────────

def plot_radar_comparison():
    """Radar (spider) chart: multi-metric comparison of top 4 systems."""
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    systems = [
        ("Hybrid V2", "nerguard_hybrid_v2_gpt-4o_nvidia-pii", C[0]),
        ("Base", "nerguard_base_nvidia-pii", C[7]),
        ("Presidio", "presidio_nvidia-pii", C[1]),
        ("Piiranha", "piiranha_nvidia-pii", C[2]),
    ]

    metrics = ["F1-macro", "F1-micro", "Entity-F1", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for name, dirname, color in systems:
        r = load_json(os.path.join(exp_dir, dirname, "results.json"))
        tl = r["token_level"]
        el = r["entity_level"]
        values = [tl["f1_macro"], tl["f1_micro"], el["f1"], el["precision"], el["recall"]]
        values += values[:1]

        ax.plot(angles, values, "-", color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.3, 0.85)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8"], fontsize=9, color="0.5")
    ax.set_title("Multi-metric System Comparison", y=1.08, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "radar_comparison.png"))
    plt.close()
    print("  [OK] radar_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Generating LaTeX tables...")
    generate_main_benchmark()
    generate_ablation_table()
    generate_cross_domain_table()
    generate_significance_table()
    generate_local_llms_table()

    print("\nGenerating plots...")
    plot_precision_recall()
    plot_pareto_frontier()
    plot_boundary_condition()
    plot_ablation_lollipop()
    plot_cross_domain_dumbbell()
    plot_significance_forest()
    plot_entity_frequency_robustness()
    plot_length_robustness()
    plot_radar_comparison()

    print(f"\nTables saved to: {TABLE_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
