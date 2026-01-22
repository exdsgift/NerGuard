import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

GLINER_AVAILABLE = False
PRESIDIO_AVAILABLE = False
SPACY_AVAILABLE = False

try:
    from gliner import GLiNER

    GLINER_AVAILABLE = True
except ImportError:
    print("⚠️  GLiNER non installato.")

try:
    from presidio_analyzer import AnalyzerEngine

    PRESIDIO_AVAILABLE = True
except ImportError:
    print("⚠️  Presidio non installato.")

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    print("⚠️  SpaCy non installato.")

try:
    from src.pipeline.entropy_inference import LLMRouter

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLMRouter non disponibile - NerGuard Hybrid disabilitato")

MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
OUTPUT_DIR = "./benchmark_results_final"
SAMPLE_LIMIT = 1000

UNIFIED_SCHEMA = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "PERSON": "PERSON",
    "PER": "PERSON",
    "CITY": "LOCATION",
    "STREET": "LOCATION",
    "ZIPCODE": "LOCATION",
    "BUILDINGNUM": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "TELEPHONENUM": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "DATE_TIME": "DATE_TIME",
    "IDCARDNUM": "GOV_ID",
    "PASSPORTNUM": "GOV_ID",
    "DRIVERLICENSENUM": "GOV_ID",
    "TAXNUM": "GOV_ID",
    "SOCIALNUM": "GOV_ID",
    "SSN": "GOV_ID",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "AGE": "AGE",
    "SEX": "GENDER",
    "GENDER": "GENDER",
    "TITLE": "TITLE",
}


@dataclass
class EntitySpan:
    label: str
    start: int
    end: int
    text: str

    @property
    def unified_label(self):
        clean = self.label.upper()
        if "-" in clean and len(clean.split("-")[0]) <= 2:
            clean = clean.split("-", 1)[1]

        if "SSN" in clean or "DRIVER" in clean or "PASSPORT" in clean:
            return "GOV_ID"
        if "MAIL" in clean:
            return "EMAIL_ADDRESS"
        if "PHONE" in clean:
            return "PHONE_NUMBER"

        return UNIFIED_SCHEMA.get(clean, "O")


def set_academic_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "axes.facecolor": "#FAFAFA",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
        }
    )


def get_model_palette(models):
    unique_models = sorted(list(set(models)))
    colors = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
        "#6A994E",
        "#BC4B51",
        "#8E7DBE",
        "#5C8001",
    ]
    return dict(zip(unique_models, colors[: len(unique_models)]))


def plot_main_metrics(df_metrics, output_dir):
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    model_colors = get_model_palette(df_metrics["Model"])

    df_f1 = df_metrics.sort_values("F1-Score")
    colors_f1 = [model_colors[m] for m in df_f1["Model"]]

    bars1 = ax1.barh(
        range(len(df_f1)),
        df_f1["F1-Score"],
        color=colors_f1,
        edgecolor="#2C2C2C",
        linewidth=1.2,
        height=0.65,
        alpha=0.9,
    )

    ax1.set_yticks(range(len(df_f1)))
    ax1.set_yticklabels(df_f1["Model"], fontsize=11)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("F1-Score", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Model Performance (F1-Score)", pad=12, fontsize=14, fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.3, linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars1, df_f1["F1-Score"])):
        ax1.text(
            val + 0.015,
            i,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    df_lat = df_metrics.sort_values("Latency (ms)", ascending=False)
    colors_lat = [model_colors[m] for m in df_lat["Model"]]

    bars2 = ax2.barh(
        range(len(df_lat)),
        df_lat["Latency (ms)"],
        color=colors_lat,
        edgecolor="#2C2C2C",
        linewidth=1.2,
        height=0.65,
        alpha=0.9,
    )

    ax2.set_yticks(range(len(df_lat)))
    ax2.set_yticklabels(df_lat["Model"], fontsize=11)
    ax2.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax2.set_title("Inference Latency", pad=12, fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, linewidth=0.8)

    max_lat = df_lat["Latency (ms)"].max()
    for i, (bar, val) in enumerate(zip(bars2, df_lat["Latency (ms)"])):
        ax2.text(
            val + (max_lat * 0.02),
            i,
            f"{val:.1f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

    plt.savefig(f"{output_dir}/main_metrics.png", facecolor="white", edgecolor="none")
    plt.close()


def plot_efficiency_frontier(df_metrics, output_dir):
    """Frontiera di efficienza con design migliorato."""
    fig, ax = plt.subplots(figsize=(11, 7))
    df_sorted = df_metrics.sort_values("Latency (ms)")

    colors_list = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
        "#6A994E",
        "#BC4B51",
        "#8E7DBE",
        "#5C8001",
    ]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        x, y = row["Latency (ms)"], row["F1-Score"]
        color = colors_list[i % len(colors_list)]
        marker = markers[i % len(markers)]

        ax.scatter(
            x,
            y,
            s=350,
            c=color,
            marker=marker,
            edgecolors="#2C2C2C",
            linewidth=2,
            alpha=0.85,
            label=row["Model"],
            zorder=5,
        )

        y_offset = -30 if y > df_sorted["F1-Score"].median() else 30
        va = "top" if y > df_sorted["F1-Score"].median() else "bottom"

        ax.annotate(
            row["Model"],
            xy=(x, y),
            xytext=(0, y_offset),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va=va,
            zorder=6,
            color="#1A1A1A",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )

    ax.set_xscale("log")
    ax.set_ylim(
        max(0, df_sorted["F1-Score"].min() - 0.08),
        min(1.05, df_sorted["F1-Score"].max() + 0.08),
    )

    ax.set_title(
        "Efficiency Frontier: Latency vs Performance",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Latency (ms) — Log Scale", fontsize=13)
    ax.set_ylabel("F1-Score", fontsize=13)
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=9.5,
        ncol=1 if len(df_sorted) <= 5 else 2,
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(f"{output_dir}/efficiency_frontier.png", facecolor="white")
    plt.close()


def plot_entity_comparison(results, output_dir):
    """Confronto prestazioni per tipo di entità con layout chiaro."""
    entity_data = []
    for res in results:
        report = classification_report(
            res["y_true"], res["y_pred"], output_dict=True, zero_division=0
        )
        for label, scores in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg", "O"]:
                entity_data.append(
                    {
                        "Model": res["model"],
                        "Entity": label,
                        "F1-Score": scores["f1-score"],
                    }
                )

    if not entity_data:
        return

    df_ent = pd.DataFrame(entity_data)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Palette
    palette = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
        "#6A994E",
        "#BC4B51",
        "#8E7DBE",
        "#5C8001",
    ]

    sns.barplot(
        data=df_ent,
        x="Entity",
        y="F1-Score",
        hue="Model",
        palette=palette,
        edgecolor="#2C2C2C",
        linewidth=1.2,
        alpha=0.9,
        ax=ax,
    )

    ax.set_title(
        "Performance Breakdown by Entity Type", fontsize=15, fontweight="bold", pad=15
    )
    ax.set_xlabel("Entity Type", fontsize=13)
    ax.set_ylabel("F1-Score", fontsize=13)
    ax.set_ylim(0, 1.05)

    ax.legend(
        title="Model",
        title_fontsize=11,
        fontsize=10,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entity_comparison.png", facecolor="white")
    plt.close()


def plot_confusion_matrix_single(y_true, y_pred, model_name, output_dir):
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])

    size = max(8, min(12, len(labels) * 0.7))
    fig, ax = plt.subplots(figsize=(size, size * 0.95))

    cmap = sns.diverging_palette(250, 10, s=80, l=55, as_cmap=True)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.5,
        linecolor="white",
        square=True,
        annot_kws={"size": 9},
        cbar_kws={"label": "Normalized Frequency", "shrink": 0.8},
        vmin=0,
        vmax=1,
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold", pad=15
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    safe_name = model_name.replace(" ", "_").replace("/", "-")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm_{safe_name}.png", facecolor="white")
    plt.close()


def plot_results(results, output_dir="evaluation_results"):
    set_academic_style()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics = []
    for res in results:
        y_true_no_o = [y for y in res["y_true"] if y != "O"]
        y_pred_no_o = [
            res["y_pred"][i] for i, y in enumerate(res["y_true"]) if y != "O"
        ]

        report = classification_report(
            y_true_no_o if y_true_no_o else res["y_true"],
            y_pred_no_o if y_true_no_o else res["y_pred"],
            output_dict=True,
            zero_division=0,
        )

        metrics.append(
            {
                "Model": res["model"],
                "F1-Score": report["macro avg"]["f1-score"],
                "Latency (ms)": res["latency"],
            }
        )

    df_metrics = pd.DataFrame(metrics)

    plot_main_metrics(df_metrics, output_dir)
    plot_efficiency_frontier(df_metrics, output_dir)
    plot_entity_comparison(results, output_dir)

    for res in results:
        plot_confusion_matrix_single(
            res["y_true"], res["y_pred"], res["model"], output_dir
        )

    df_metrics.to_csv(f"{output_dir}/metrics_summary.csv", index=False)


class ModelWrapper:
    def predict(self, sample, tokenizer, id2label) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class NerGuard(ModelWrapper):
    def __init__(self, model_path, enable_llm=True):
        self.enable_llm = enable_llm and LLM_AVAILABLE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(
            self.device
        )
        self.model.eval()

        self.router = LLMRouter(source="openai") if self.enable_llm else None
        self.id2label = self.model.config.id2label

        self.thresh_ent = 0.583
        self.thresh_conf = 0.787

        self.llm_calls = 0
        self.llm_changed_to_entity = 0
        self.llm_changed_to_o = 0
        self.llm_kept_same = 0
        self.high_entropy_tokens = 0

        if self.enable_llm:
            print(
                f"    LLM enabled with thresholds: entropy > {self.thresh_ent}, conf < {self.thresh_conf}"
            )

    def name(self):
        return "NerGuard (Hybrid)" if self.enable_llm else "NerGuard (Base)"

    def predict(self, sample, tokenizer, id2label):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decodifica testo per LLM
        text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Ottieni offset mapping (CRITICO per LLM)
        try:
            encoding = tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]
        except:
            encoding = tokenizer.encode_plus(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]

        # Predizione modello
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids_tensor, attention_mask=attention_mask).logits[
                0
            ]

        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
        conf, pred_ids = torch.max(probs, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        final_preds = []
        final_gt = []
        prev_pred = "O"

        for i, lbl_id in enumerate(labels):
            if lbl_id == -100:
                continue

            gt_lbl = id2label[lbl_id].replace("B-", "").replace("I-", "").upper()
            final_gt.append(UNIFIED_SCHEMA.get(gt_lbl, "O"))
            pred_id = pred_ids[i].item()
            raw_pred = self.id2label[pred_id]

            if (
                self.enable_llm
                and entropy[i] > self.thresh_ent
                and conf[i] < self.thresh_conf
            ):
                self.high_entropy_tokens += 1

                try:
                    self.llm_calls += 1

                    if i < len(offset_mapping):
                        char_start, char_end = offset_mapping[i]

                        # DEBUG
                        if self.llm_calls <= 3:
                            print(f"\n  [LLM DEBUG #{self.llm_calls}]")
                            print(
                                f"    Token: '{tokens[i]}' | Entropy: {entropy[i]:.3f} | Conf: {conf[i]:.3f}"
                            )
                            print(f"    Pred: {raw_pred} | Prev: {prev_pred}")
                            print(
                                f"    Chars: {char_start}-{char_end} in '{text[max(0, char_start - 20) : char_end + 20]}'"
                            )

                        res = self.router.disambiguate(
                            target_token=tokens[i],
                            full_text=text,
                            char_start=char_start,
                            char_end=char_end,
                            current_pred=raw_pred,
                            prev_label=prev_pred,
                            lang="en",
                        )

                        # Track changes
                        old_is_entity = raw_pred != "O"
                        new_label = res.get("corrected_label", raw_pred)
                        new_is_entity = new_label != "O"

                        if self.llm_calls <= 3:
                            print(
                                f"    LLM Result: {new_label} (is_pii: {res.get('is_pii')})"
                            )

                        if old_is_entity == new_is_entity:
                            self.llm_kept_same += 1
                        elif not old_is_entity and new_is_entity:
                            self.llm_changed_to_entity += 1
                        elif old_is_entity and not new_is_entity:
                            self.llm_changed_to_o += 1

                        if res.get("is_pii"):
                            raw_pred = new_label
                        else:
                            raw_pred = "O"
                except Exception as e:
                    if self.llm_calls <= 3:
                        print(f"    ERROR: {e}")
                    pass

            clean_pred = raw_pred.replace("B-", "").replace("I-", "").upper()
            final_preds.append(UNIFIED_SCHEMA.get(clean_pred, "O"))
            prev_pred = raw_pred

        return final_preds, final_gt


class ExternalWrapper(ModelWrapper):
    def __init__(self, engine_type):
        self.type = engine_type
        if engine_type == "spacy" and SPACY_AVAILABLE:
            if not spacy.util.is_package("en_core_web_lg"):
                spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
        elif engine_type == "presidio" and PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
        elif engine_type == "gliner" and GLINER_AVAILABLE:
            self.model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.labels = [
                "person",
                "city",
                "street",
                "phone",
                "email",
                "date",
                "id",
                "passport",
                "age",
                "gender",
                "job",
            ]
            self.map = {
                "person": "PERSON",
                "city": "LOCATION",
                "street": "LOCATION",
                "phone": "PHONE_NUMBER",
                "email": "EMAIL_ADDRESS",
                "date": "DATE_TIME",
                "id": "GOV_ID",
                "passport": "GOV_ID",
                "age": "AGE",
                "gender": "GENDER",
                "job": "TITLE",
            }

    def name(self):
        return self.type.capitalize()

    def _get_spans(self, text):
        spans = []
        if self.type == "spacy":
            doc = self.nlp(text)
            spans = [
                EntitySpan(ent.label_, ent.start_char, ent.end_char, ent.text)
                for ent in doc.ents
            ]
        elif self.type == "presidio":
            res = self.analyzer.analyze(text=text, language="en")
            spans = [
                EntitySpan(r.entity_type, r.start, r.end, text[r.start : r.end])
                for r in res
            ]
        elif self.type == "gliner":
            preds = self.model.predict_entities(text, self.labels, threshold=0.5)
            spans = [
                EntitySpan(
                    self.map.get(p["label"], "O"), p["start"], p["end"], p["text"]
                )
                for p in preds
            ]
        return spans

    def predict(self, sample, tokenizer, id2label):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        spans = self._get_spans(text)

        try:
            encoding = tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]
        except:
            encoding = tokenizer.encode_plus(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]

        final_preds = []
        final_gt = []

        for i, lbl_id in enumerate(labels):
            if lbl_id == -100:
                continue

            gt_lbl = id2label[lbl_id].replace("B-", "").replace("I-", "").upper()
            final_gt.append(UNIFIED_SCHEMA.get(gt_lbl, "O"))

            if i < len(offset_mapping):
                start_char, end_char = offset_mapping[i]

                if start_char == end_char == 0:
                    final_preds.append("O")
                    continue

                token_pred = "O"
                for span in spans:
                    overlap_start = max(start_char, span.start)
                    overlap_end = min(end_char, span.end)

                    if overlap_end > overlap_start:
                        overlap_ratio = (overlap_end - overlap_start) / (
                            end_char - start_char
                        )
                        if overlap_ratio > 0.5:
                            token_pred = span.unified_label
                            break

                final_preds.append(token_pred)
            else:
                final_preds.append("O")

        return final_preds, final_gt


# main


def run_benchmark():
    print(f"Loading Data from {DATA_PATH}...")
    dataset = load_from_disk(DATA_PATH)["validation"]
    if SAMPLE_LIMIT:
        dataset = dataset.select(range(min(SAMPLE_LIMIT, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    id2label = AutoConfig.from_pretrained(MODEL_PATH).id2label

    models = []
    models.append(NerGuard(MODEL_PATH, enable_llm=False))

    if LLM_AVAILABLE:
        models.append(NerGuard(MODEL_PATH, enable_llm=True))
    else:
        print("   - Skipping NerGuard Hybrid (LLMRouter not available)\n")

    if SPACY_AVAILABLE:
        models.append(ExternalWrapper("spacy"))
    if PRESIDIO_AVAILABLE:
        models.append(ExternalWrapper("presidio"))
    if GLINER_AVAILABLE:
        models.append(ExternalWrapper("gliner"))

    results = []
    print(f"- Benchmarking {len(models)} models...\n")

    for model in models:
        print(f"- {model.name()}")
        y_true, y_pred, lats = [], [], []

        for sample in tqdm(dataset, desc=f"  Processing", leave=False):
            t0 = time.time()
            try:
                p, g = model.predict(sample, tokenizer, id2label)

                if len(p) == len(g) and len(p) > 0:
                    y_pred.extend(p)
                    y_true.extend(g)
                    lats.append((time.time() - t0) * 1000)
            except Exception as e:
                pass

        avg_lat = np.mean(lats) if lats else 0

        llm_info = ""
        if isinstance(model, NerGuard) and model.enable_llm:
            total_tokens = len(
                [l for sample in dataset for l in sample["labels"] if l != -100]
            )
            llm_info = f"\n    High entropy tokens: {model.high_entropy_tokens}/{total_tokens} ({100 * model.high_entropy_tokens / total_tokens:.1f}%)"
            llm_info += f"\n    LLM calls: {model.llm_calls}"
            llm_info += f"\n    Changes: O→Entity={model.llm_changed_to_entity}, Entity→O={model.llm_changed_to_o}, Same={model.llm_kept_same}"

        print(f"  ✓ {len(lats)} samples | {avg_lat:.2f}ms{llm_info}\n")

        results.append(
            {
                "model": model.name(),
                "y_true": y_true,
                "y_pred": y_pred,
                "latency": avg_lat,
            }
        )

    plot_results(results)
    print(f"Done! Results in {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    run_benchmark()
