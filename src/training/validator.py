"""
Validation script for NerGuard hybrid model.

This script evaluates the model comparing baseline (DeBERTa only) vs hybrid
(DeBERTa + LLM routing) performance.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from datasets import load_from_disk
from tqdm import tqdm

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.core.model_loader import load_model_and_tokenizer, get_device
from src.core.metrics import compute_entropy_confidence
from src.inference.llm_router import LLMRouter
from src.utils.io import ensure_dir
from src.utils.logging_config import setup_logging

logger = setup_logging("ModelValidation")


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    baseline_preds: np.ndarray
    hybrid_preds: np.ndarray
    true_labels: np.ndarray
    correct_entropies: np.ndarray
    incorrect_entropies: np.ndarray
    llm_interventions: int
    llm_corrections: int


def process_sample(
    sample: Dict,
    model,
    tokenizer,
    router: Optional[LLMRouter],
    id2label: Dict,
    label2id: Dict,
    device: torch.device,
    entropy_threshold: float,
    confidence_threshold: float,
    use_llm: bool = True,
) -> Tuple:
    """Process a single sample and return predictions + metrics."""

    input_ids = torch.tensor([sample["input_ids"]]).to(device)
    attention_mask = torch.tensor([sample["attention_mask"]]).to(device)
    labels = sample["labels"]

    # Decode text for LLM context
    full_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    # Get offset mapping
    if "offset_mapping" in sample and sample["offset_mapping"] is not None:
        offsets = sample["offset_mapping"]
    else:
        tokenized_ref = tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = tokenized_ref["offset_mapping"]

    tokens_str = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    special_token_ids = set(tokenizer.all_special_ids)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0]
    entropy, confidences, pred_ids = compute_entropy_confidence(logits)

    # Convert to numpy
    pred_ids_np = pred_ids.cpu().numpy()
    conf_np = confidences.cpu().numpy()
    entr_np = entropy.cpu().numpy()

    # Collect results
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

        # Skip special tokens and ignored labels
        if lbl == -100 or token_id in special_token_ids:
            continue

        base_pred_id = pred_ids_np[j]
        conf = conf_np[j]
        ent = entr_np[j]

        baseline_preds_list.append(base_pred_id)
        true_labels_list.append(lbl)

        # Track entropy for correct/incorrect predictions
        if base_pred_id == lbl:
            correct_entropies_list.append(float(ent))
        else:
            incorrect_entropies_list.append(float(ent))

        hybrid_pred_id = base_pred_id
        curr_base_label = id2label[base_pred_id]

        # Get character offsets
        char_start, char_end = 0, 0
        if offset_idx < len(offsets):
            char_start, char_end = offsets[offset_idx]
            offset_idx += 1

        # LLM routing decision
        should_trigger = use_llm and router and ent > entropy_threshold and conf < confidence_threshold

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


def evaluate(
    model_path: str = DEFAULT_MODEL_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = "./results/validation",
    llm_routing: bool = True,
    sample_limit: Optional[int] = 1000,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    llm_source: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
):
    """
    Run hybrid evaluation comparing baseline vs hybrid model.

    Args:
        model_path: Path to trained model
        data_path: Path to dataset
        output_dir: Directory to save results
        llm_routing: Whether to enable LLM routing
        sample_limit: Maximum samples to evaluate (None for all)
        entropy_threshold: Entropy threshold for LLM routing
        confidence_threshold: Confidence threshold for LLM routing
    """
    logger.info("HYBRID EVALUATION PROTOCOL")
    logger.info(f"Model: {model_path}")
    ensure_dir(output_dir)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load model and tokenizer
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path, device=str(device))

    # Initialize LLM router
    router = None
    if llm_routing:
        import os
        source = llm_source or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")
        try:
            if source == "openai":
                router = LLMRouter(source="openai", model=openai_model)
            else:
                router = LLMRouter(source="ollama")
            logger.info(f"LLM Router: {source} ({openai_model if source == 'openai' else 'default'})")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM router: {e}")
            llm_routing = False

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(data_path)
    eval_dataset = dataset["validation"]

    if sample_limit:
        logger.warning(f"Limiting to {sample_limit} samples")
        eval_dataset = eval_dataset.select(range(min(sample_limit, len(eval_dataset))))

    id2label = model.config.id2label
    label2id = model.config.label2id

    # Collect predictions
    baseline_preds = []
    hybrid_preds = []
    true_labels = []
    correct_entropies = []
    incorrect_entropies = []
    llm_intervention_total = 0
    llm_correction_total = 0

    logger.info("Running inference...")
    for i in tqdm(range(len(eval_dataset)), desc="Processing"):
        sample = eval_dataset[i]

        results = process_sample(
            sample, model, tokenizer, router, id2label, label2id,
            device, entropy_threshold, confidence_threshold, llm_routing
        )

        base_p, hyb_p, true_l, corr_e, incorr_e, interv, corr = results

        baseline_preds.extend(base_p)
        hybrid_preds.extend(hyb_p)
        true_labels.extend(true_l)
        correct_entropies.extend(corr_e)
        incorrect_entropies.extend(incorr_e)
        llm_intervention_total += interv
        llm_correction_total += corr

        # Clear cache periodically
        if i % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert to numpy
    baseline_preds = np.array(baseline_preds)
    hybrid_preds = np.array(hybrid_preds)
    true_labels = np.array(true_labels)
    correct_entropies = np.array(correct_entropies)
    incorrect_entropies = np.array(incorrect_entropies)

    # Print report
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Tokens: {len(true_labels)}")
    logger.info(f"LLM Intervention Rate: {llm_intervention_total / len(true_labels):.2%}")
    logger.info(f"LLM Corrections: {llm_correction_total}/{llm_intervention_total}")
    logger.info("=" * 60)

    active_labels = sorted(list(set(true_labels)))
    target_names = [id2label[i] for i in active_labels]

    # Classification reports
    base_report_dict = classification_report(
        true_labels, baseline_preds,
        labels=active_labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    hybrid_report_dict = classification_report(
        true_labels, hybrid_preds,
        labels=active_labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    logger.info("\nBASELINE (DeBERTa only)")
    logger.info(classification_report(
        true_labels, baseline_preds,
        labels=active_labels, target_names=target_names,
        digits=4, zero_division=0,
    ))

    logger.info("\nHYBRID (DeBERTa + LLM)")
    logger.info(classification_report(
        true_labels, hybrid_preds,
        labels=active_labels, target_names=target_names,
        digits=4, zero_division=0,
    ))

    # Save text report
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write("BASELINE PERFORMANCE\n")
        f.write(classification_report(
            true_labels, baseline_preds,
            labels=active_labels, target_names=target_names,
            digits=4, zero_division=0,
        ))
        f.write("\n\nHYBRID PERFORMANCE\n")
        f.write(classification_report(
            true_labels, hybrid_preds,
            labels=active_labels, target_names=target_names,
            digits=4, zero_division=0,
        ))
        f.write(f"\n\nSTATS\n")
        f.write(f"Total Tokens: {len(true_labels)}\n")
        f.write(f"LLM Triggers: {llm_intervention_total}\n")
        f.write(f"LLM Corrections: {llm_correction_total}\n")
        f.write(f"Intervention Rate: {llm_intervention_total / len(true_labels):.2%}\n")

    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    evaluate(llm_routing=True)
