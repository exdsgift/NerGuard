"""
Regex-based PII validator for NerGuard.

Provides two capabilities:
  1. can_skip_llm(entity_class, text, char_start, char_end)
       — Before LLM routing: if regex confirms the predicted entity, skip LLM call.
  2. correct_predictions(text, offset_mapping, preds, id2label, label2id)
       — Post-processing: promote O-predicted tokens to entity labels where
         regex finds a PII pattern that the model missed.

Supported entity classes and their patterns:
  CREDITCARDNUMBER  — major card formats, Luhn check
  EMAIL             — RFC 5322 simplified
  SOCIALNUM         — US SSN (XXX-XX-XXXX)
  IBAN              — ISO 13616, check-digit validation
  TELEPHONENUM      — E.164 and common national formats
  DATE              — ISO 8601, DD/MM/YYYY, MM/DD/YYYY
  ZIPCODE           — US ZIP, UK postcode

Design principles:
  - High precision over recall (false positives are harmful, false negatives are tolerable)
  - Luhn/checksum validation where possible
  - Plug-and-play: entities without a pattern are unaffected
  - Patterns are conservative; ambiguous matches are not promoted
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validators (checksum / structural checks)
# ---------------------------------------------------------------------------

def _luhn_check(number_str: str) -> bool:
    """Return True if the digit sequence passes the Luhn algorithm."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    digits.reverse()
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_check(raw: str) -> bool:
    """Return True if the string is a structurally valid IBAN (ISO 13616)."""
    iban = raw.replace(" ", "").replace("-", "").upper()
    if not (15 <= len(iban) <= 34):
        return False
    if not iban[:2].isalpha() or not iban[2:4].isdigit():
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = ""
    for c in rearranged:
        if c.isdigit():
            numeric += c
        elif "A" <= c <= "Z":
            numeric += str(ord(c) - ord("A") + 10)
        else:
            return False
    try:
        return int(numeric) % 97 == 1
    except (ValueError, OverflowError):
        return False


def _ssn_check(raw: str) -> bool:
    """Return True for a plausibly valid US SSN (basic structural check)."""
    digits = [c for c in raw if c.isdigit()]
    if len(digits) != 9:
        return False
    # All-zero groups are invalid
    if digits[:3] == ["0"] * 3:
        return False
    if digits[3:5] == ["0"] * 2:
        return False
    if digits[5:] == ["0"] * 4:
        return False
    return True


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

@dataclass
class RegexConfig:
    pattern: str
    validator_fn: Optional[Callable[[str], bool]] = None
    enabled: bool = True


# Each key is a model entity class (without BIO prefix).
# Patterns are designed for high precision — they require the canonical format.
REGEX_PATTERNS: Dict[str, RegexConfig] = {
    # Credit card numbers — major networks, optional space/dash separators, Luhn check
    "CREDITCARDNUMBER": RegexConfig(
        pattern=(
            r"\b(?:"
            r"4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"   # Visa 16
            r"|5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # MC
            r"|2(?:2[2-9][1-9]|[3-6][0-9]{2}|7[01][0-9]|720)[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # MC 2-series
            r"|3[47][0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{5}"              # Amex 15
            r"|3(?:0[0-5]|[68][0-9])[0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{4}"  # Diners 14
            r"|6(?:011|5[0-9]{2})[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # Discover
            r")\b"
        ),
        validator_fn=_luhn_check,
    ),

    # Email — simplified RFC 5322
    "EMAIL": RegexConfig(
        pattern=r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        validator_fn=None,
    ),

    # US Social Security Number
    "SOCIALNUM": RegexConfig(
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        validator_fn=_ssn_check,
    ),

    # IBAN — country code + check digits + BBAN
    "IBAN": RegexConfig(
        pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
        validator_fn=_iban_check,
    ),

    # Phone numbers — E.164 and common formats (conservative: require min 7 digits)
    "TELEPHONENUM": RegexConfig(
        pattern=(
            r"\b(?:"
            r"\+?\d{1,3}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{1,4})?"
            r"|\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"   # (NXX) NXX-XXXX
            r")\b"
        ),
        validator_fn=None,
    ),

    # Dates — ISO 8601, US, European
    "DATE": RegexConfig(
        pattern=(
            r"\b(?:"
            r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"       # ISO YYYY-MM-DD
            r"|(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{2,4}"    # MM/DD/YY(YY)
            r"|(?:0[1-9]|[12]\d|3[01])[./](?:0[1-9]|1[0-2])[./]\d{2,4}"  # DD.MM.YY(YY)
            r")\b"
        ),
        validator_fn=None,
    ),

    # ZIP / Postal codes — US 5-digit (±4), UK basic
    "ZIPCODE": RegexConfig(
        pattern=(
            r"\b(?:"
            r"\d{5}(?:-\d{4})?"                   # US ZIP / ZIP+4
            r"|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}"  # UK postcode
            r")\b"
        ),
        validator_fn=None,
    ),
}


# ---------------------------------------------------------------------------
# RegexValidator
# ---------------------------------------------------------------------------

class RegexValidator:
    """
    Post-processing validator that uses regex patterns to:
      - Skip LLM routing when regex already confirms the model prediction
      - Promote O-predicted tokens to entity labels where regex finds PII

    Usage:
        validator = RegexValidator()

        # Before LLM call (routing guard)
        if validator.can_skip_llm("CREDITCARDNUMBER", text, char_start, char_end):
            continue  # skip LLM

        # After full hybrid pipeline (post-processing)
        corrected = validator.correct_predictions(
            text, offset_mapping, hybrid_preds, id2label, label2id
        )
    """

    def __init__(self, patterns: Optional[Dict[str, RegexConfig]] = None):
        self.patterns = patterns or REGEX_PATTERNS
        self._compiled: Dict[str, re.Pattern] = {}
        for cls, cfg in self.patterns.items():
            if cfg.enabled:
                self._compiled[cls] = re.compile(cfg.pattern, re.IGNORECASE)

        self.stats: Dict[str, int] = defaultdict(int)

    def can_skip_llm(
        self,
        entity_class: str,
        text: str,
        char_start: int,
        char_end: int,
    ) -> bool:
        """
        Return True if regex confirms that the token span [char_start, char_end)
        belongs to entity_class, meaning the LLM routing call can be skipped.

        The token must be fully contained within a regex match that also passes
        the optional validator function.
        """
        compiled = self._compiled.get(entity_class)
        if compiled is None:
            return False

        config = self.patterns[entity_class]
        for match in compiled.finditer(text):
            if match.start() <= char_start and match.end() >= char_end:
                if config.validator_fn and not config.validator_fn(match.group()):
                    continue
                self.stats[f"skip_llm_{entity_class}"] += 1
                return True
        return False

    def correct_predictions(
        self,
        text: str,
        offset_mapping: np.ndarray,
        preds: np.ndarray,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
    ) -> np.ndarray:
        """
        Post-process token-level predictions: promote contiguous O-predicted
        tokens to entity labels where a regex match is found.

        Only promotes tokens that are ALL predicted as O (avoids overwriting
        partial entity predictions from the model or LLM router).

        Returns a copy of `preds` with any corrections applied.
        """
        corrected = preds.copy()
        o_id = label2id.get("O", 0)

        for entity_class, compiled in self._compiled.items():
            config = self.patterns[entity_class]
            for match in compiled.finditer(text):
                if config.validator_fn and not config.validator_fn(match.group()):
                    continue

                # Collect token indices overlapping this match
                match_tokens: List[int] = []
                for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start == tok_end:  # special token ([CLS], [SEP], padding)
                        continue
                    if tok_end <= match.start() or tok_start >= match.end():
                        continue
                    match_tokens.append(idx)

                if not match_tokens:
                    continue

                # Only promote if ALL overlapping tokens are currently O
                if not all(corrected[i] == o_id for i in match_tokens):
                    continue

                for i, tok_idx in enumerate(match_tokens):
                    bio_prefix = "B-" if i == 0 else "I-"
                    label_str = f"{bio_prefix}{entity_class}"
                    label_id = label2id.get(label_str)
                    if label_id is None:
                        break  # entity class not in model vocabulary — skip
                    corrected[tok_idx] = label_id

                self.stats[f"promoted_{entity_class}"] += len(match_tokens)
                logger.debug(
                    f"Regex promoted {len(match_tokens)} tokens to {entity_class}: "
                    f"'{match.group()}'"
                )

        return corrected

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        self.stats.clear()
