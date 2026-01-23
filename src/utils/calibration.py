"""
Calibration and debugging utilities for NerGuard.

This module provides tools for debugging token-label alignment and model predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.utils.colors import Colors
from src.core.metrics import compute_entropy_confidence

logger = logging.getLogger(__name__)


def check_data_alignment(
    sample: Dict,
    tokenizer: PreTrainedTokenizer,
    id2label: Dict[int, str],
    max_tokens: int = 50,
) -> None:
    """
    Debug function to visualize token-label alignment.

    Prints tokens with their corresponding labels, highlighting entities.

    Args:
        sample: Dataset sample with 'input_ids' and 'labels'
        tokenizer: Tokenizer for decoding tokens
        id2label: Mapping from label IDs to label names
        max_tokens: Maximum number of tokens to display
    """
    input_ids = sample["input_ids"][:max_tokens]
    labels = sample["labels"][:max_tokens]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n" + "=" * 60)
    print("TOKEN-LABEL ALIGNMENT CHECK")
    print("=" * 60)

    for i, (token, label_id) in enumerate(zip(tokens, labels)):
        if label_id == -100:
            label_str = "[IGNORED]"
            color = Colors.DIM
        else:
            label_str = id2label.get(label_id, "UNK")
            if label_str == "O":
                color = Colors.ENDC
            elif label_str.startswith("B-"):
                color = Colors.OKGREEN
            else:
                color = Colors.OKCYAN

        print(f"{i:3d}: {color}{token:<20} -> {label_str}{Colors.ENDC}")


def analyze_predictions(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    id2label: Dict[int, str],
    device: torch.device,
    show_all: bool = False,
) -> List[Dict[str, Any]]:
    """
    Analyze model predictions for a given text.

    Args:
        text: Input text to analyze
        model: Trained NER model
        tokenizer: Tokenizer
        id2label: Label mapping
        device: Torch device
        show_all: If True, show all tokens. If False, only show entities.

    Returns:
        List of dictionaries with prediction details
    """
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    input_ids = encoding["input_ids"].to(device)
    offsets = encoding["offset_mapping"][0].tolist()

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]

    entropy, confidence, predictions = compute_entropy_confidence(logits)
    probs = torch.softmax(logits, dim=-1)

    # Get second best predictions
    top2 = torch.topk(probs, k=2, dim=-1)
    second_best_probs = top2.values[:, 1]
    second_best_ids = top2.indices[:, 1]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    results = []

    print("\n" + "=" * 80)
    print(f"{'TOKEN':<20} {'PRED':<15} {'CONF':>8} {'ENTR':>8} {'2ND BEST':<15}")
    print("=" * 80)

    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        pred_id = predictions[i].item()
        pred_label = id2label.get(pred_id, "UNK")
        conf = confidence[i].item()
        ent = entropy[i].item()

        second_id = second_best_ids[i].item()
        second_label = id2label.get(second_id, "UNK")
        second_prob = second_best_probs[i].item()
        margin = conf - second_prob

        result = {
            "token": token,
            "prediction": pred_label,
            "confidence": conf,
            "entropy": ent,
            "second_best": second_label,
            "margin": margin,
            "offset": offsets[i] if i < len(offsets) else (0, 0),
        }
        results.append(result)

        # Color coding
        if pred_label != "O":
            color = Colors.OKGREEN if conf > 0.8 else Colors.WARNING
        elif conf < 0.5:
            color = Colors.FAIL
        else:
            color = Colors.ENDC

        if show_all or pred_label != "O":
            second_info = f"{second_label} ({second_prob:.3f})"
            print(
                f"{color}{token:<20} {pred_label:<15} {conf:>8.4f} {ent:>8.4f} {second_info:<15}{Colors.ENDC}"
            )

    return results


def debug_decode_entities(
    text: str,
    results: List[Dict[str, Any]],
) -> str:
    """
    Reconstruct text with highlighted entities.

    Args:
        text: Original text
        results: Prediction results from analyze_predictions

    Returns:
        Text with ANSI color highlighting for entities
    """
    # Sort by offset
    sorted_results = sorted(
        [r for r in results if r["prediction"] != "O"],
        key=lambda x: x["offset"][0],
        reverse=True,
    )

    highlighted = text
    for result in sorted_results:
        start, end = result["offset"]
        if start == end:
            continue

        entity_text = text[start:end]
        label = result["prediction"]
        colored = f"{Colors.OKGREEN}[{entity_text}|{label}]{Colors.ENDC}"
        highlighted = highlighted[:start] + colored + highlighted[end:]

    return highlighted


def generate_classification_report(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    id2label: Dict[int, str],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a classification report on a dataset.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        id2label: Label mapping
        device: Torch device
        max_samples: Maximum samples to evaluate

    Returns:
        Dictionary with classification metrics
    """
    from sklearn.metrics import classification_report

    model.eval()

    all_preds = []
    all_labels = []

    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))

    for sample in samples:
        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        labels = sample["labels"]

        with torch.no_grad():
            outputs = model(input_ids)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Filter out ignored labels
        for pred, label in zip(preds, labels):
            if label != -100:
                all_preds.append(pred)
                all_labels.append(label)

    # Generate report
    target_names = [id2label.get(i, f"UNK_{i}") for i in sorted(set(all_labels))]

    report = classification_report(
        all_labels,
        all_preds,
        labels=sorted(set(all_labels)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Print human-readable report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=sorted(set(all_labels)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    return report
