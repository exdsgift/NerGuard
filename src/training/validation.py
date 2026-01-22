import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import os
import sys
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()
from src.pipeline.optimize_llmrouting import OptimizedLLMRouter  # noqa: E402

# Configuration
MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
OUTPUT_DIR = "./validation_results"
THRESHOLD_ENTROPY = 0.583
THRESHOLD_CONF = 0.787
SAMPLE_LIMIT = 1000
BATCH_SIZE = 16


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    baseline_preds: np.ndarray
    hybrid_preds: np.ndarray
    true_labels: np.ndarray
    correct_entropies: np.ndarray
    incorrect_entropies: np.ndarray
    llm_interventions: int
    llm_corrections: int


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict, title: str, filename: str
):
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    label_names = [id2label[i] for i in unique_labels]
    n_labels = len(unique_labels)

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Normalizzazione robusta
    cm_norm = np.divide(
        cm.astype("float"),
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1, keepdims=True) != 0,
    )

    # Dimensioni adattive con limite superiore
    plot_w = min(max(12, n_labels * 0.6), 30)
    plot_h = min(max(10, n_labels * 0.5), 25)

    plt.figure(figsize=(plot_w, plot_h))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 9},
        cbar_kws={"label": "Recall"},
    )

    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.title(title, pad=20, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved: {filename}")


def plot_entropy_separation(
    correct_entropies: np.ndarray,
    incorrect_entropies: np.ndarray,
    threshold: float,
    filename: str,
):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        correct_entropies, fill=True, color="green", label="Correct", clip=(0, 2.0)
    )
    sns.kdeplot(
        incorrect_entropies, fill=True, color="red", label="Incorrect", clip=(0, 2.0)
    )

    plt.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )

    plt.title("Entropy Distribution (Uncertainty Analysis)")
    plt.xlabel("Shannon Entropy")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(0, 1.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved: {filename}")


def plot_model_comparison(baseline_report: Dict, hybrid_report: Dict, filename: str):
    labels = [
        k
        for k in baseline_report.keys()
        if k not in ["accuracy", "macro avg", "weighted avg"] and k in hybrid_report
    ]

    base_f1 = [baseline_report[k]["f1-score"] for k in labels]
    hybrid_f1 = [hybrid_report[k]["f1-score"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(max(12, len(labels) * 0.4), 6))

    plt.bar(
        x - width / 2,
        base_f1,
        width,
        label="Baseline",
        color="lightgray",
        edgecolor="black",
    )
    plt.bar(
        x + width / 2,
        hybrid_f1,
        width,
        label="Hybrid (Ours)",
        color="#4c72b0",
        edgecolor="black",
    )

    plt.xlabel("Class")
    plt.ylabel("F1-Score")
    plt.title("Baseline vs Hybrid Performance")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend(loc="lower right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved: {filename}")


def process_sample(
    sample: Dict,
    model,
    tokenizer,
    router,
    id2label: Dict,
    label2id: Dict,
    device: torch.device,
    use_llm: bool = True,
) -> Tuple:
    """Process a single sample and return predictions + metrics"""

    input_ids = torch.tensor([sample["input_ids"]]).to(device)
    attention_mask = torch.tensor([sample["attention_mask"]]).to(device)
    labels = sample["labels"]

    full_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    # FIXED: Usa offset_mapping se già disponibile
    if "offset_mapping" in sample and sample["offset_mapping"] is not None:
        offsets = sample["offset_mapping"]
    else:
        tokenized_ref = tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = tokenized_ref["offset_mapping"]

    tokens_str = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    special_token_ids = set(tokenizer.all_special_ids)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0]
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    entropy = -torch.sum(probs * log_probs, dim=-1)
    confidences, pred_ids = torch.max(probs, dim=-1)

    # Convert to numpy once
    pred_ids_np = pred_ids.cpu().numpy()
    conf_np = confidences.cpu().numpy()
    entr_np = entropy.cpu().numpy()

    baseline_preds_list = []
    hybrid_preds_list = []
    true_labels_list = []
    correct_entropies_list = []
    incorrect_entropies_list = []

    llm_intervention_count = 0
    llm_correction_count = 0

    prev_label_hybrid = "O"
    offset_idx = 0

    for j in range(len(labels)):
        lbl = labels[j]
        token_id = sample["input_ids"][j]
        token_str = tokens_str[j]

        if lbl == -100 or token_id in special_token_ids:
            continue

        base_pred_id = pred_ids_np[j]
        conf = conf_np[j]
        ent = entr_np[j]

        baseline_preds_list.append(base_pred_id)
        true_labels_list.append(lbl)

        if base_pred_id == lbl:
            correct_entropies_list.append(float(ent))
        else:
            incorrect_entropies_list.append(float(ent))

        hybrid_pred_id = base_pred_id
        curr_base_label = id2label[base_pred_id]

        char_start, char_end = 0, 0
        if offset_idx < len(offsets):
            char_start, char_end = offsets[offset_idx]
            offset_idx += 1
        else:
            logger.warning(
                f"Offset index {offset_idx} exceeds offsets length {len(offsets)}"
            )

        should_trigger = use_llm and ent > THRESHOLD_ENTROPY and conf < THRESHOLD_CONF

        if should_trigger:
            llm_intervention_count += 1
            try:
                llm_result = router.disambiguate(
                    target_token=token_str,
                    full_text=full_text,
                    char_start=char_start,
                    char_end=char_end,
                    current_pred=curr_base_label,
                    prev_label=prev_label_hybrid,
                    lang="en",
                )

                if llm_result.get("is_pii"):
                    corrected_label = llm_result.get("corrected_label", "O")

                    if corrected_label in label2id:
                        proposed_id = label2id[corrected_label]
                        hybrid_pred_id = proposed_id

                        if proposed_id != base_pred_id:
                            llm_correction_count += 1

            except Exception as e:
                logger.error(f"LLM error at token '{token_str}': {str(e)}")

        hybrid_preds_list.append(hybrid_pred_id)
        prev_label_hybrid = id2label[hybrid_pred_id]

    return (
        baseline_preds_list,
        hybrid_preds_list,
        true_labels_list,
        correct_entropies_list,
        incorrect_entropies_list,
        llm_intervention_count,
        llm_correction_count,
    )


def evaluate(llm_routing: bool = True):
    logger.info("HYBRID EVALUATION PROTOCOL")
    logger.info(f"Model Source: {MODEL_PATH}")
    ensure_dir(OUTPUT_DIR)
    set_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    router = OptimizedLLMRouter(source="openai") if llm_routing else None

    logger.info("Loading Validation Dataset...")
    dataset = load_from_disk(DATA_PATH)
    eval_dataset = dataset["validation"]

    if SAMPLE_LIMIT:
        logger.warning(f"TEST MODE: Limiting to first {SAMPLE_LIMIT} samples.")
        eval_dataset = eval_dataset.select(range(min(SAMPLE_LIMIT, len(eval_dataset))))

    id2label = model.config.id2label
    label2id = model.config.label2id

    estimated_tokens = len(eval_dataset) * 100  # Rough estimate
    baseline_preds = []
    hybrid_preds = []
    true_labels = []
    correct_entropies = []
    incorrect_entropies = []

    llm_intervention_total = 0
    llm_correction_total = 0

    logger.info("Running Inference Loop...")

    for i in tqdm(range(len(eval_dataset)), desc="Processing"):
        sample = eval_dataset[i]

        results = process_sample(
            sample, model, tokenizer, router, id2label, label2id, device, llm_routing
        )

        base_p, hyb_p, true_l, corr_e, incorr_e, interv, corr = results

        baseline_preds.extend(base_p)
        hybrid_preds.extend(hyb_p)
        true_labels.extend(true_l)
        correct_entropies.extend(corr_e)
        incorrect_entropies.extend(incorr_e)
        llm_intervention_total += interv
        llm_correction_total += corr

        if i % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert to numpy arrays
    baseline_preds = np.array(baseline_preds)
    hybrid_preds = np.array(hybrid_preds)
    true_labels = np.array(true_labels)
    correct_entropies = np.array(correct_entropies)
    incorrect_entropies = np.array(incorrect_entropies)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Tokens Evaluated: {len(true_labels)}")
    logger.info(
        f"LLM Intervention Rate:  {llm_intervention_total / len(true_labels):.2%}"
    )
    logger.info(
        f"LLM Correction Yield:   {llm_correction_total}/{llm_intervention_total} triggers changed label"
    )
    logger.info("=" * 60)

    active_labels = sorted(list(set(true_labels)))
    target_names = [id2label[i] for i in active_labels]

    base_report_dict = classification_report(
        true_labels,
        baseline_preds,
        labels=active_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    hybrid_report_dict = classification_report(
        true_labels,
        hybrid_preds,
        labels=active_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    logger.info("\nDeBERTa v3 (Baseline)")
    logger.info(
        classification_report(
            true_labels,
            baseline_preds,
            labels=active_labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    logger.info("\nDeBERTa + LLM (Hybrid)")
    logger.info(
        classification_report(
            true_labels,
            hybrid_preds,
            labels=active_labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    logger.info("\nGenerating Plots...")

    plot_confusion_matrix(
        true_labels,
        baseline_preds,
        id2label,
        "Baseline Confusion Matrix",
        f"{OUTPUT_DIR}/cm_baseline.png",
    )
    plot_confusion_matrix(
        true_labels,
        hybrid_preds,
        id2label,
        "Hybrid Confusion Matrix",
        f"{OUTPUT_DIR}/cm_hybrid.png",
    )
    plot_entropy_separation(
        correct_entropies,
        incorrect_entropies,
        THRESHOLD_ENTROPY,
        f"{OUTPUT_DIR}/entropy_calibration.png",
    )
    plot_model_comparison(
        base_report_dict, hybrid_report_dict, f"{OUTPUT_DIR}/f1_comparison.png"
    )

    # Save detailed report
    with open(f"{OUTPUT_DIR}/report.txt", "w") as f:
        f.write("BASELINE PERFORMANCE\n")
        f.write(
            classification_report(
                true_labels,
                baseline_preds,
                labels=active_labels,
                target_names=target_names,
                digits=4,
                zero_division=0,
            )
        )
        f.write("\n\nHYBRID PERFORMANCE\n")
        f.write(
            classification_report(
                true_labels,
                hybrid_preds,
                labels=active_labels,
                target_names=target_names,
                digits=4,
                zero_division=0,
            )
        )
        f.write(f"\n\nSTATS\n")
        f.write(f"Total Tokens: {len(true_labels)}\n")
        f.write(f"LLM Triggers: {llm_intervention_total}\n")
        f.write(f"LLM Corrections: {llm_correction_total}\n")
        f.write(f"Intervention Rate: {llm_intervention_total / len(true_labels):.2%}\n")

    logger.info(f"\nEvaluation complete! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    evaluate(llm_routing=True)
