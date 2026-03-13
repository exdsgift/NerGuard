"""Regenerate all thesis plots with unified SciencePlots academic style.

Uses style combinations optimized per plot type:
- Scatter plots: ['science', 'nature', 'scatter']
- Bar/line charts: ['science', 'nature']

Usage:
    uv run python -m src.scripts.regenerate_thesis_plots
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import scienceplots  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "Gabriele_Durante_TdL", "plots", "scienceplots")

# Fixed subplot margins so that the axes rectangle is identical across all plots.
# This ensures consistent plot-area size when imported into LaTeX.
_MARGINS = dict(left=0.15, right=0.97, top=0.95, bottom=0.15)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def _annotate_no_overlap(ax, pts, fontsize=5, xscale="linear"):
    """Place scatter annotations in evenly-spaced y-slots, fully inside axes.

    Labels are stacked vertically in the right margin of the plot (or left
    for points on the far right), with thin leader lines to the data point.
    """
    if not pts:
        return

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    n = len(pts)

    sorted_pts = sorted(pts, key=lambda p: p[1])

    # Evenly-spaced y-slots spanning central 80 % of y-range
    margin = 0.10
    slot_lo = ylim[0] + y_range * margin
    slot_hi = ylim[1] - y_range * margin
    slot_step = (slot_hi - slot_lo) / max(n - 1, 1)
    targets_y = [slot_lo + i * slot_step for i in range(n)]

    # Compute text x position: fixed fraction into the axes from the side
    if xscale == "log":
        xlo, xhi = math.log10(max(xlim[0], 1e-10)), math.log10(max(xlim[1], 1e-10))
        tx_right = 10 ** (xlo + (xhi - xlo) * 0.98)
        tx_left = 10 ** (xlo + (xhi - xlo) * 0.02)
    else:
        x_range = xlim[1] - xlim[0]
        tx_right = xlim[1] - x_range * 0.02
        tx_left = xlim[0] + x_range * 0.02

    for rank, (x, y, text) in enumerate(sorted_pts):
        if xscale == "log":
            lx = math.log10(max(x, 1e-10))
            rel_x = (lx - xlo) / (xhi - xlo) if xhi != xlo else 0.5
        else:
            rel_x = (x - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5

        if rel_x < 0.55:
            tx, ha = tx_right, "right"
        else:
            tx, ha = tx_left, "left"

        ty = targets_y[rank]
        ax.annotate(text, (x, y), xytext=(tx, ty), textcoords="data",
                    fontsize=fontsize, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.3, alpha=0.4))


# ── Plot 1: Boundary condition (scatter) ─────────────────────────────────────

def plot_boundary_condition():
    data = load_json(os.path.join(ROOT, "plots", "boundary_n_classes_analysis.json"))
    log_n = np.log(data["n_classes"])
    deltas = np.array(data["deltas"])
    fit = data["linear_fit"]

    with plt.style.context(["science", "nature", "scatter"]):
        fig, ax = plt.subplots()
        ax.scatter(log_n, deltas)
        for d, x, y in zip(data["domains"], log_n, deltas):
            ax.annotate(d, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6)

        x_fit = np.linspace(min(log_n) - 0.5, max(log_n) + 0.5, 100)
        ax.plot(x_fit, fit["slope"] * x_fit + fit["intercept"], "--",
                label=r"$\Delta = %.3f \ln(n) + %.3f$" % (fit["slope"], fit["intercept"]))
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel(r"$\ln$(Number of Entity Classes)")
        ax.set_ylabel(r"$\Delta$ Entity-F1 (Hybrid $-$ Base)")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend()
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("boundary_condition.pdf"))
        plt.close(fig)
    print("  [1/7] boundary_condition.pdf")


# ── Plot 2: Error decomposition (bar) ────────────────────────────────────────

def plot_error_decomposition():
    data = load_json(os.path.join(
        ROOT, "experiments", "error_decomposition_2026-03-09", "results.json"))

    categories = ["missed_detection", "boundary_error", "type_confusion"]
    labels = ["Missed detection", "Boundary error", "Type confusion"]
    base = [data["base"]["error_counts"].get(c, 0) for c in categories]
    hybrid = [data["hybrid"]["error_counts"].get(c, 0) for c in categories]

    y = np.arange(len(categories))
    h = 0.35

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.barh(y + h / 2, base, h, label="Base")
        ax.barh(y - h / 2, hybrid, h, label="Hybrid")
        for i in range(len(categories)):
            ax.text(base[i] + 3, y[i] + h / 2, str(base[i]), va="center", fontsize=6)
            ax.text(hybrid[i] + 3, y[i] - h / 2, str(hybrid[i]), va="center", fontsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Number of False Negative Spans")
        ax.legend()
        fig.subplots_adjust(left=0.28, right=0.97, top=0.95, bottom=0.15)
        fig.savefig(_out("error_decomposition.pdf"))
        plt.close(fig)
    print("  [2/7] error_decomposition.pdf")


# ── Plot 3: Reliability diagram (bar + line) ─────────────────────────────────

def plot_reliability_diagram():
    data = load_json(os.path.join(
        ROOT, "experiments", "calibration_2026-03-09", "results.json"))
    bins = data["reliability_diagram_pre"]
    ece = data["ece_pre"]

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=0.5,
                label="Perfect calibration")
        mids = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        ax.bar(mids, accs, width=0.08, label="Accuracy")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Observed Accuracy")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.text(0.95, 0.05, r"ECE $= %.4f$" % ece, transform=ax.transAxes,
                ha="right", fontsize=7)
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("reliability_diagram.pdf"))
        plt.close(fig)
    print("  [3/7] reliability_diagram.pdf")


# ── Plot 4: F1 vs Latency (scatter) ──────────────────────────────────────────

def plot_f1_vs_latency():
    # Load both cloud and local experiment results
    cloud_path = os.path.join(
        ROOT, "experiments", "2026-03-03_13-23", "summary", "comparison_table.json")
    local_path = os.path.join(
        ROOT, "experiments", "local_1000samples_2026-03-03", "summary", "comparison_table.json")

    subset = []
    for p in [cloud_path, local_path]:
        if os.path.exists(p):
            subset.extend(r for r in load_json(p) if r["dataset"] == "nvidia-pii")
    if not subset:
        print("  [4/7] SKIPPED")
        return

    short_map = {
        "NerGuard Base": "NerGuard Base", "NerGuard Hybrid (gpt-4o)": "Hybrid V1",
        "NerGuard Hybrid V2 (gpt-4o)": "Hybrid V2 (gpt-4o)", "Piiranha": "Piiranha",
        "Presidio": "Presidio", "spaCy (en_core_web_trf)": "spaCy",
        "dslim/bert-base-NER": "BERT-NER",
    }
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "d", "H", "<", ">"]

    with plt.style.context(["science", "nature", "scatter"]):
        fig, ax = plt.subplots()
        pts = []
        for i, r in enumerate(sorted(subset, key=lambda x: x.get("f1_macro", 0), reverse=True)):
            short = short_map.get(r["system"], r["system"])
            if "Hybrid V2 (" in r["system"] and r["system"] not in short_map:
                short = "V2 (" + r["system"].split("Hybrid V2 (")[1].rstrip(")") + ")"
            ax.scatter(r["latency_mean_ms"], r["f1_macro"],
                       marker=markers[i % len(markers)], label=short, zorder=3)
            pts.append((r["latency_mean_ms"], r["f1_macro"], short))

        ax.set_xscale("log")
        _annotate_no_overlap(ax, pts, fontsize=4, xscale="log")
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_ylabel("F1-macro")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend(fontsize=4, loc="lower right", ncol=2)
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("f1_vs_latency_nvidia-pii.pdf"))
        plt.close(fig)
    print("  [4/7] f1_vs_latency_nvidia-pii.pdf")


# ── Plot 5: Precision vs Recall (scatter) ────────────────────────────────────

def plot_precision_recall():
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    systems = [
        ("NerGuard Hybrid V2", "nerguard_hybrid_v2_gpt-4o_nvidia-pii"),
        ("Presidio", "presidio_nvidia-pii"),
        ("NerGuard Hybrid V1", "nerguard_hybrid_gpt-4o_nvidia-pii"),
        ("Piiranha", "piiranha_nvidia-pii"),
        ("NerGuard Base", "nerguard_base_nvidia-pii"),
        ("spaCy", "spacy_en_core_web_trf_nvidia-pii"),
        ("dslim/BERT", "dslim_bert-base-ner_nvidia-pii"),
    ]
    markers = ["o", "s", "D", "^", "p", "v", "X"]

    with plt.style.context(["science", "nature", "scatter"]):
        fig, ax = plt.subplots()

        for f1_val in [0.4, 0.5, 0.6, 0.7, 0.8]:
            p = np.linspace(f1_val / 2 + 0.001, 1.0, 200)
            r = (f1_val * p) / (2 * p - f1_val)
            mask = (r > 0) & (r <= 1) & (p > 0.3)
            ax.plot(r[mask], p[mask], "-", color="0.85", linewidth=0.4, zorder=1)

        pts = []
        for i, (name, dirname) in enumerate(systems):
            rpath = os.path.join(exp_dir, dirname, "results.json")
            if not os.path.exists(rpath):
                continue
            el = load_json(rpath)["entity_level"]
            ax.scatter(el["recall"], el["precision"], marker=markers[i],
                       label=name, zorder=5)
            pts.append((el["recall"], el["precision"], name))

        _annotate_no_overlap(ax, pts, fontsize=4)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlim(0.50, 0.85)
        ax.set_ylim(0.38, 0.72)
        ax.legend(loc="lower left", ncol=2, fontsize=4)
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("precision_recall_nvidia-pii.pdf"))
        plt.close(fig)
    print("  [5/7] precision_recall_nvidia-pii.pdf")


# ── Plot 6: Training loss (line) ─────────────────────────────────────────────

def plot_training_loss():
    state = load_json(os.path.join(
        ROOT, "models", "mdeberta-pii-safe", "checkpoint-21759", "trainer_state.json"))
    log = state["log_history"]

    train = [e for e in log if "loss" in e and "eval_loss" not in e]
    steps = [e["step"] for e in train]
    losses = [e["loss"] for e in train]
    evals = [e for e in log if "eval_loss" in e]

    ema = []
    s = losses[0]
    for v in losses:
        s = 0.85 * s + 0.15 * v
        ema.append(s)

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(steps, losses, linewidth=0.3, alpha=0.4, label="Training loss (raw)")
        ax.plot(steps, ema, linewidth=0.8, label="Training loss (EMA)")
        ax.scatter([e["step"] for e in evals], [e["eval_loss"] for e in evals],
                   marker="D", s=15, zorder=5, label="Validation loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend()
        ax.set_ylim(bottom=0)
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("training_loss.pdf"))
        plt.close(fig)
    print("  [6/7] training_loss.pdf")


# ── Plot 7: Eval F1 (line) ───────────────────────────────────────────────────

def plot_eval_f1():
    state = load_json(os.path.join(
        ROOT, "models", "mdeberta-pii-safe", "checkpoint-21759", "trainer_state.json"))
    evals = [e for e in state["log_history"] if "eval_f1" in e]
    epochs = [e["epoch"] for e in evals]

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(epochs, [e["eval_f1"] for e in evals], "-", label="F1")
        ax.plot(epochs, [e["eval_precision"] for e in evals], "-", label="Precision")
        ax.plot(epochs, [e["eval_recall"] for e in evals], "-", label="Recall")

        for ep, f1 in zip(epochs, [e["eval_f1"] for e in evals]):
            ax.annotate(f"{f1:.4f}", (ep, f1), textcoords="offset points",
                        xytext=(0, 6), fontsize=5, ha="center")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend()
        ax.set_ylim(0.90, 0.97)
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("eval_f1.pdf"))
        plt.close(fig)
    print("  [7/7] eval_f1.pdf")


# ── Plot 8: Routing Efficiency (scatter) ──────────────────────────────────────

def plot_routing_efficiency():
    data = load_json(os.path.join(
        ROOT, "experiments", "ablation_nvidia-pii_1000samples", "ablation_summary.json"))

    labels = {
        "never_route": "Never route",
        "always_route": "Always route",
        "entropy_only": "Entropy only",
        "confidence_only": "Confidence only",
        "full_system": "Full system",
    }
    markers = ["v", "^", "s", "D", "o"]

    with plt.style.context(["science", "nature", "scatter"]):
        fig, ax = plt.subplots()
        for i, v in enumerate(data):
            name = labels.get(v["variant"], v["variant"])
            ax.scatter(v["routing"]["llm_calls"], v["entity_level"]["f1"],
                       marker=markers[i], label=name, s=30, zorder=5)
            ax.annotate(name, (v["routing"]["llm_calls"], v["entity_level"]["f1"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=5)

        ax.set_xlabel("Number of LLM Calls")
        ax.set_ylabel("Entity-F1")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend(fontsize=5)
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("routing_efficiency.pdf"))
        plt.close(fig)
    print("  [8/13] routing_efficiency.pdf")


# ── Plot 9: Per-Document Delta Distribution (violin) ─────────────────────────

def plot_delta_distribution():
    data = load_json(os.path.join(
        ROOT, "experiments", "significance_hierarchical_2026-03-09", "results.json"))

    domain_labels = []
    all_deltas = []
    p_values = []
    for d in data:
        short = d["domain"].split(" (")[0]
        domain_labels.append(short)
        all_deltas.append(d["hierarchical_bootstrap"]["per_doc_deltas"])
        p_values.append(d["hierarchical_bootstrap"]["p_value"])

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        parts = ax.violinplot(all_deltas, positions=range(len(domain_labels)),
                              showmeans=True, showmedians=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)

        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(domain_labels)))
        ax.set_xticklabels(domain_labels, fontsize=7)
        ax.set_ylabel(r"$\Delta$ Entity-F1 per Document")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        for i, p in enumerate(p_values):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ymax = max(all_deltas[i])
            ax.text(i, ymax + 0.03, f"p{star}", ha="center", fontsize=6)

        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("delta_distribution.pdf"))
        plt.close(fig)
    print("  [9/13] delta_distribution.pdf")


# ── Plot 10: Per-Entity Error Reduction Heatmap ──────────────────────────────

def plot_error_heatmap():
    data = load_json(os.path.join(
        ROOT, "experiments", "error_decomposition_2026-03-09", "results.json"))

    error_types = ["missed_detection", "type_confusion", "boundary_error"]
    error_labels = ["Missed det.", "Type conf.", "Boundary err."]

    all_entities = sorted(set(
        list(data["base"]["per_entity_errors"].keys()) +
        list(data["hybrid"]["per_entity_errors"].keys())
    ))

    base_mat = np.zeros((len(all_entities), len(error_types)))
    hybrid_mat = np.zeros((len(all_entities), len(error_types)))
    for i, ent in enumerate(all_entities):
        for j, et in enumerate(error_types):
            base_mat[i, j] = data["base"]["per_entity_errors"].get(ent, {}).get(et, 0)
            hybrid_mat[i, j] = data["hybrid"]["per_entity_errors"].get(ent, {}).get(et, 0)

    diff = base_mat - hybrid_mat  # positive = hybrid reduced errors

    # Filter out entities with no change
    mask = np.any(diff != 0, axis=1)
    diff = diff[mask]
    entities_filtered = [e.replace("_", " ").title() for e, m in zip(all_entities, mask) if m]

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        im = ax.imshow(diff, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(error_labels)))
        ax.set_xticklabels(error_labels, fontsize=6)
        ax.set_yticks(range(len(entities_filtered)))
        ax.set_yticklabels(entities_filtered, fontsize=5)

        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                val = int(diff[i, j])
                if val != 0:
                    color = "white" if abs(val) > diff.max() * 0.6 else "black"
                    sign = "+" if val > 0 else ""
                    ax.text(j, i, f"{sign}{val}", ha="center", va="center",
                            fontsize=5, color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Error Reduction (Base - Hybrid)", fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        fig.subplots_adjust(left=0.25, right=0.90, top=0.95, bottom=0.08)
        fig.savefig(_out("error_reduction_heatmap.pdf"))
        plt.close(fig)
    print("  [10/13] error_reduction_heatmap.pdf")


# ── Plot 11: Cross-Domain Performance (grouped bar) ──────────────────────────

def plot_cross_domain():
    domains = [
        ("PII", "nvidia_pii_500samples_2026-03-05"),
        ("BC5CDR", "bc5cdr_500samples_2026-03-05"),
        ("BUSTER", "buster_500samples_2026-03-05"),
    ]
    sig_data = load_json(os.path.join(
        ROOT, "experiments", "significance_hierarchical_2026-03-09", "results.json"))
    sig_map = {}
    for s in sig_data:
        short = s["domain"].split(" (")[0]
        sig_map[short] = s["hierarchical_bootstrap"]["p_value"]

    base_f1s, hybrid_f1s, labels = [], [], []
    for name, dirname in domains:
        table = load_json(os.path.join(ROOT, "experiments", dirname, "summary",
                                       "comparison_table.json"))
        base = [r for r in table if "Base" in r["system"]][0]
        hybrid = [r for r in table if "Hybrid" in r["system"]][0]
        base_f1s.append(base["entity_f1"])
        hybrid_f1s.append(hybrid["entity_f1"])
        labels.append(name)

    x = np.arange(len(labels))
    w = 0.35

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.bar(x - w / 2, base_f1s, w, label="Base")
        ax.bar(x + w / 2, hybrid_f1s, w, label="Hybrid")

        for i, name in enumerate(labels):
            p = sig_map.get(name, 1.0)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ymax = max(base_f1s[i], hybrid_f1s[i])
            ax.text(i, ymax + 0.008, star, ha="center", fontsize=7, style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Entity-F1")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylim(0.55, 0.95)
        ax.legend()
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("cross_domain_performance.pdf"))
        plt.close(fig)
    print("  [11/13] cross_domain_performance.pdf")


# ── Plot 12: Length Bucket Performance (grouped bar) ──────────────────────────

def plot_length_buckets():
    exp_dir = os.path.join(ROOT, "experiments", "2026-03-03_13-23")
    base = load_json(os.path.join(exp_dir, "nerguard_base_nvidia-pii", "results.json"))
    hybrid = load_json(os.path.join(
        exp_dir, "nerguard_hybrid_v2_gpt-4o_nvidia-pii", "results.json"))

    buckets = ["short", "medium", "long"]
    bucket_labels = ["Short", "Medium", "Long"]
    base_f1 = [base["per_length_bucket"][b]["f1_macro"] for b in buckets]
    hybrid_f1 = [hybrid["per_length_bucket"][b]["f1_macro"] for b in buckets]
    n_samples = [base["per_length_bucket"][b]["n_samples"] for b in buckets]

    x = np.arange(len(buckets))
    w = 0.35

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        bars_b = ax.bar(x - w / 2, base_f1, w, label="Base")
        bars_h = ax.bar(x + w / 2, hybrid_f1, w, label="Hybrid")

        for i in range(len(buckets)):
            ax.text(i - w / 2, base_f1[i] + 0.01, f"{base_f1[i]:.2f}",
                    ha="center", fontsize=5)
            ax.text(i + w / 2, hybrid_f1[i] + 0.01, f"{hybrid_f1[i]:.2f}",
                    ha="center", fontsize=5)
            ax.text(i, -0.04, f"n={n_samples[i]}", ha="center", fontsize=5,
                    transform=ax.get_xaxis_transform())

        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels)
        ax.set_ylabel("F1-macro")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylim(0, 0.90)
        ax.legend()
        fig.subplots_adjust(**_MARGINS)
        fig.savefig(_out("length_bucket_performance.pdf"))
        plt.close(fig)
    print("  [12/13] length_bucket_performance.pdf")


# ── Plot 13: Local LLM Comparison (bar) ──────────────────────────────────────

def plot_local_llm():
    data = load_json(os.path.join(
        ROOT, "experiments", "local_1000samples_2026-03-03", "summary",
        "comparison_table.json"))

    # Also load cloud gpt-4o for reference
    cloud = load_json(os.path.join(
        ROOT, "experiments", "2026-03-03_13-23", "summary", "comparison_table.json"))
    gpt4o = [r for r in cloud if r["system"] == "NerGuard Hybrid V2 (gpt-4o)"]
    base = [r for r in cloud if r["system"] == "NerGuard Base"]

    all_models = sorted(data, key=lambda x: x["entity_f1"], reverse=True)
    if gpt4o:
        all_models.insert(0, gpt4o[0])

    names = []
    f1s = []
    latencies = []
    for r in all_models:
        name = r["system"].replace("NerGuard Hybrid V2 (", "").rstrip(")")
        names.append(name)
        f1s.append(r["entity_f1"])
        latencies.append(r["latency_mean_ms"])

    with plt.style.context(["science", "nature"]):
        fig, ax1 = plt.subplots()

        x = np.arange(len(names))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]
        bars = ax1.bar(x, f1s, 0.6, color=bar_colors, zorder=3)
        # Highlight gpt-4o bar
        if gpt4o:
            bars[0].set_edgecolor("black")
            bars[0].set_linewidth(1.0)

        # Add base F1 reference line
        if base:
            ax1.axhline(y=base[0]["entity_f1"], color="red", linewidth=0.7,
                        linestyle="--", label=f"Base ({base[0]['entity_f1']:.3f})", zorder=2)

        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
        ax1.set_ylabel("Entity-F1")
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_ylim(0.60, 0.67)

        # Overlay latency on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, latencies, "-", color="gray", linewidth=0.8,
                 label="Latency", zorder=4)
        ax2.set_ylabel("Mean Latency (ms)", fontsize=7)
        ax2.set_yscale("log")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="upper right")

        fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.22)
        fig.savefig(_out("local_llm_comparison.pdf"))
        plt.close(fig)
    print("  [13/13] local_llm_comparison.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Output: {OUT_DIR}\n")
    plot_boundary_condition()
    plot_error_decomposition()
    plot_reliability_diagram()
    plot_f1_vs_latency()
    plot_precision_recall()
    plot_training_loss()
    plot_eval_f1()
    plot_routing_efficiency()
    plot_delta_distribution()
    plot_error_heatmap()
    plot_cross_domain()
    plot_length_buckets()
    plot_local_llm()
    print("\nDone.")


if __name__ == "__main__":
    main()
