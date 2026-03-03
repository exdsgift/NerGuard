"""
Span assembler for NerGuard — groups consecutive B-X / I-X tokens into entity spans.

Shared between the hybrid evaluator (training-time evaluation) and the benchmark
system wrapper (cross-system comparison). Extracted to avoid code duplication.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.inference.entity_router import EntitySpecificRouter


@dataclass
class EntitySpan:
    """A contiguous span of B-X / I-X tokens of the same entity class."""

    indices: List[int]  # token indices in the flat sequence
    entity_class: str  # entity class without BIO prefix, e.g. "SURNAME"
    is_uncertain: bool  # True if the B- anchor token meets routing thresholds
    char_start: int  # char start of the first token
    char_end: int  # char end of the last token


def assemble_entity_spans(
    pred_labels: List[str],
    entropy_flat,
    conf_flat,
    offset_flat,
    entity_router: EntitySpecificRouter,
    labels_flat=None,
) -> List[EntitySpan]:
    """
    Group consecutive B-X / I-X predicted tokens into EntitySpan objects.

    Rules:
    - B-X starts a new span; is_uncertain = entity_router.should_route(B-X, ...)
    - Following I-X tokens of the same class are appended to the span
    - A -100 token (excluded or special) breaks the span (only when labels_flat provided)
    - O tokens and orphan I- tokens produce no span (skipped)

    The routing decision is anchored on the B- token: if the B- is confident,
    the entire span is skipped (anchor propagation). This eliminates the harmful
    per-token routing of continuation tokens (I-SURNAME, I-GIVENNAME, etc.).

    Args:
        pred_labels: BIO labels predicted by the model, e.g. ["O", "B-SURNAME", "I-SURNAME"]
        entropy_flat: Per-token entropy values
        conf_flat: Per-token confidence values
        offset_flat: Per-token (char_start, char_end) offset tuples
        entity_router: Router that decides whether to route based on uncertainty
        labels_flat: Optional ground truth label IDs. When provided, tokens with
            value -100 break span assembly (used in evaluation with excluded labels).
            When None, no tokens are skipped (used in inference/benchmark).
    """
    spans: List[EntitySpan] = []
    n = len(pred_labels)
    i = 0
    while i < n:
        # Skip excluded tokens (only when ground truth labels are available)
        if labels_flat is not None and labels_flat[i] == -100:
            i += 1
            continue

        label = pred_labels[i]

        if label.startswith("B-"):
            entity_class = label[2:]
            is_uncertain = entity_router.should_route(
                predicted_label=label,
                entropy=float(entropy_flat[i]),
                confidence=float(conf_flat[i]),
            )
            indices = [i]
            char_end = int(offset_flat[i][1])

            j = i + 1
            while j < n:
                if labels_flat is not None and labels_flat[j] == -100:
                    break
                if pred_labels[j] != f"I-{entity_class}":
                    break
                indices.append(j)
                char_end = int(offset_flat[j][1])
                j += 1

            spans.append(
                EntitySpan(
                    indices=indices,
                    entity_class=entity_class,
                    is_uncertain=is_uncertain,
                    char_start=int(offset_flat[i][0]),
                    char_end=char_end,
                )
            )
            i = j

        else:
            # O or orphan I- token: no span
            i += 1

    return spans


def assemble_uncertain_o_spans(
    pred_labels: List[str],
    entropy_flat,
    conf_flat,
    offset_flat,
    entropy_threshold: float,
    confidence_threshold: float,
    entropy_multiplier: float = 1.5,
    min_span_chars: int = 2,
) -> List[EntitySpan]:
    """
    Find uncertain O-predicted tokens and group consecutive ones into candidate spans.

    These are tokens the NER model labeled as non-entity but with high uncertainty,
    suggesting they might be missed PII (false negatives). Routed to LLM for
    classification with a broader label vocabulary.

    Args:
        pred_labels: BIO labels predicted by the model
        entropy_flat: Per-token entropy values
        conf_flat: Per-token confidence values
        offset_flat: Per-token (char_start, char_end) offset tuples
        entropy_threshold: Base entropy threshold
        confidence_threshold: Base confidence threshold
        entropy_multiplier: Multiplier for O-token entropy threshold (default 1.5x)
        min_span_chars: Minimum character length for a candidate span
    """
    o_thresh_entropy = entropy_threshold * entropy_multiplier
    o_thresh_confidence = confidence_threshold

    n = len(pred_labels)
    spans: List[EntitySpan] = []
    i = 0

    while i < n:
        label = pred_labels[i]
        start_off = offset_flat[i]

        # Skip non-O tokens and special tokens (offset 0,0)
        if label != "O" or (int(start_off[0]) == 0 and int(start_off[1]) == 0):
            i += 1
            continue

        ent = float(entropy_flat[i])
        conf = float(conf_flat[i])

        if ent > o_thresh_entropy and conf < o_thresh_confidence:
            # Start a candidate span — group consecutive uncertain O tokens
            indices = [i]
            char_end = int(offset_flat[i][1])

            j = i + 1
            while j < n:
                if pred_labels[j] != "O":
                    break
                sj = offset_flat[j]
                if int(sj[0]) == 0 and int(sj[1]) == 0:
                    break
                ej = float(entropy_flat[j])
                cj = float(conf_flat[j])
                if ej > o_thresh_entropy and cj < o_thresh_confidence:
                    indices.append(j)
                    char_end = int(sj[1])
                    j += 1
                else:
                    break

            char_start = int(offset_flat[indices[0]][0])
            if char_end - char_start >= min_span_chars:
                spans.append(
                    EntitySpan(
                        indices=indices,
                        entity_class="O",
                        is_uncertain=True,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
            i = j
        else:
            i += 1

    return spans
