"""Token-span alignment: convert character-level entity spans to BIO token labels.

This module does NOT perform any label mapping or renaming.
System-native labels are preserved as-is in the output BIO sequence.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CharSpan:
    """A character-level entity span with its native label."""

    label: str  # Native label from the system (e.g., PERSON, CREDIT_CARD)
    start: int  # Character start offset
    end: int  # Character end offset
    text: str = ""


def align_spans_to_tokens(
    entity_spans: List[CharSpan],
    tokens: List[str],
    token_spans: List[Tuple[int, int]],
    overlap_threshold: float = 0.5,
) -> List[str]:
    """Convert character-level entity spans to BIO token labels.

    For each token, finds the entity span with maximum overlap.
    If overlap_ratio >= threshold, assigns the entity's native label.
    First token in an entity gets B-{label}, subsequent tokens get I-{label}.

    Args:
        entity_spans: Character-level spans from a system, with native labels.
        tokens: Word-level tokens.
        token_spans: Character (start, end) offsets per token.
        overlap_threshold: Minimum overlap ratio to assign a label.

    Returns:
        List of BIO labels, one per token. Non-entity tokens get "O".
    """
    labels = ["O"] * len(tokens)

    if not entity_spans:
        return labels

    # Sort spans by start position for deterministic assignment
    sorted_spans = sorted(entity_spans, key=lambda s: (s.start, s.end))

    # Track which span each token belongs to, for BIO assignment
    token_span_assignments = [None] * len(tokens)

    for tok_idx, (tok_start, tok_end) in enumerate(token_spans):
        if tok_start == tok_end:
            continue

        tok_len = tok_end - tok_start
        best_overlap = 0.0
        best_span_idx = None

        for span_idx, span in enumerate(sorted_spans):
            # No overlap possible
            if span.end <= tok_start or span.start >= tok_end:
                continue

            overlap_start = max(tok_start, span.start)
            overlap_end = min(tok_end, span.end)
            overlap_ratio = (overlap_end - overlap_start) / tok_len

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_span_idx = span_idx

        if best_span_idx is not None and best_overlap >= overlap_threshold:
            token_span_assignments[tok_idx] = best_span_idx

    # Assign BIO labels based on span assignments
    for tok_idx, span_idx in enumerate(token_span_assignments):
        if span_idx is None:
            continue

        span = sorted_spans[span_idx]
        # Determine B- vs I-: first token in this span gets B-, rest get I-
        if tok_idx == 0 or token_span_assignments[tok_idx - 1] != span_idx:
            labels[tok_idx] = f"B-{span.label}"
        else:
            labels[tok_idx] = f"I-{span.label}"

    # BIO repair: ensure no I- without preceding B- of same type
    for i in range(len(labels)):
        if labels[i].startswith("I-"):
            entity_type = labels[i][2:]
            if i == 0:
                labels[i] = f"B-{entity_type}"
            else:
                prev = labels[i - 1]
                prev_type = prev[2:] if prev.startswith(("B-", "I-")) else None
                if prev_type != entity_type:
                    labels[i] = f"B-{entity_type}"

    return labels


def word_tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Simple whitespace-based word tokenizer that tracks character offsets.

    Args:
        text: Input text.

    Returns:
        Tuple of (tokens, token_spans) where each span is (start, end).
    """
    tokens = []
    spans = []
    i = 0
    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        # Collect token
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        tokens.append(text[start:i])
        spans.append((start, i))
    return tokens, spans
