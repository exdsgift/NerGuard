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


def _credit_card_check(number_str: str) -> bool:
    """Return True if this looks like a credit card number.

    Network-prefixed numbers (Visa 4, MC 51-55/2221-2720, Amex 34/37,
    Discover 6011/65, Diners 300-305/36/38) are accepted with only a
    digit-count check (13-19 digits). This handles synthetic datasets
    where Luhn checksums may not be valid.

    Numbers without a recognized prefix must pass the Luhn algorithm.
    """
    digits = [c for c in number_str if c.isdigit()]
    n = len(digits)
    if not (13 <= n <= 19):
        return False

    prefix = "".join(digits[:4])
    d1 = int(digits[0])
    d2 = int("".join(digits[:2]))
    d4 = int(prefix)

    # Visa: starts with 4
    if d1 == 4 and n == 16:
        return True
    # Mastercard: 51-55 or 2221-2720
    if (51 <= d2 <= 55 and n == 16) or (2221 <= d4 <= 2720 and n == 16):
        return True
    # Amex: 34 or 37
    if d2 in (34, 37) and n == 15:
        return True
    # Discover: 6011 or 65
    if (d4 == 6011 or d2 == 65) and n == 16:
        return True
    # Diners: 300-305, 36, 38
    d3 = int("".join(digits[:3]))
    if (300 <= d3 <= 305 or d2 in (36, 38)) and n == 14:
        return True

    # No recognized prefix — require Luhn
    return _luhn_check(number_str)


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
    force_override: bool = False  # Override ANY prediction (not just O)


# Each key is a model entity class (without BIO prefix).
# Patterns are designed for high precision — they require the canonical format.
REGEX_PATTERNS: Dict[str, RegexConfig] = {
    # Credit card numbers — network-prefixed + permissive (any 13-19 digit Luhn-valid sequence)
    # force_override=True: the validator (_credit_card_check) guarantees high precision,
    # so override even when the model predicted a different entity (e.g. SOCIALNUM).
    "CREDITCARDNUMBER": RegexConfig(
        pattern=(
            r"(?:"
            # Network-prefixed patterns (high confidence)
            r"\b(?:"
            r"4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"   # Visa 16
            r"|5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # MC
            r"|2(?:2[2-9][1-9]|[3-6][0-9]{2}|7[01][0-9]|720)[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # MC 2-series
            r"|3[47][0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{5}"              # Amex 15
            r"|3(?:0[0-5]|[68][0-9])[0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{4}"  # Diners 14
            r"|6(?:011|5[0-9]{2})[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # Discover
            r")\b"
            # Permissive: any 13-19 digit sequence with separators, Luhn-validated
            r"|\b(?:\d[\s\-.]?){12,18}\d\b"
            r")"
        ),
        validator_fn=_credit_card_check,
        force_override=True,
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

    # Passport numbers — common formats (conservative, require letter prefix)
    "PASSPORTNUM": RegexConfig(
        pattern=(
            r"\b(?:"
            r"[A-Z]{1,2}\d{6,9}"       # US/UK/EU: 1-2 letters + 6-9 digits
            r"|\d{2}[A-Z]{2}\d{5}"     # Some European formats
            r")\b"
        ),
        validator_fn=None,
    ),

    # Tax ID numbers — US EIN and EU VAT (conservative)
    "TAXNUM": RegexConfig(
        pattern=(
            r"\b(?:"
            r"\d{2}-\d{7}"              # US EIN: XX-XXXXXXX
            r"|[A-Z]{2}\d{8,11}"        # EU VAT: country code + 8-11 digits
            r")\b"
        ),
        validator_fn=None,
    ),

    # Driver's license — US formats (1 letter + 6-14 digits, conservative)
    "DRIVERLICENSENUM": RegexConfig(
        pattern=r"\b[A-Z]\d{6,14}\b",
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
        correct_partial: bool = False,
    ) -> np.ndarray:
        """
        Post-process token-level predictions using regex patterns.

        Two modes:
          - Default (correct_partial=False): Only promote ALL-O token spans
            to entity labels where a validated regex match is found.
          - Aggressive (correct_partial=True): Also repair broken BIO
            sequences within validated regex matches.  If any token in the
            match already has the correct entity type, the whole span is
            overwritten with proper B-/I- sequencing.  This fixes cases like
            ``B-SOCIALNUM O I-SOCIALNUM`` → ``B-SOCIALNUM I-SOCIALNUM I-SOCIALNUM``.

        Returns a copy of ``preds`` with any corrections applied.
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

                all_o = all(corrected[i] == o_id for i in match_tokens)

                if all_o:
                    # Standard promotion: all tokens are O → assign proper BIO
                    self._apply_bio_sequence(
                        corrected, match_tokens, entity_class, label2id
                    )
                    self.stats[f"promoted_{entity_class}"] += len(match_tokens)
                    logger.debug(
                        f"Regex promoted {len(match_tokens)} tokens to {entity_class}: "
                        f"'{match.group()}'"
                    )
                elif config.force_override:
                    # Force override: validator-confirmed match overrides ANY
                    # prediction (e.g. SOCIALNUM → CREDITCARDNUMBER).
                    already_correct = all(
                        id2label.get(int(corrected[i]), "O").endswith(f"-{entity_class}")
                        for i in match_tokens
                    )
                    if not already_correct:
                        self._apply_bio_sequence(
                            corrected, match_tokens, entity_class, label2id
                        )
                        self.stats[f"overridden_{entity_class}"] += len(match_tokens)
                        logger.debug(
                            f"Regex force-overrode {len(match_tokens)} tokens to {entity_class}: "
                            f"'{match.group()}'"
                        )
                elif correct_partial:
                    # Aggressive repair: fix broken BIO spans within the match.
                    # Only apply if at least one token already has the correct
                    # entity type (avoids overwriting unrelated predictions).
                    has_correct_entity = any(
                        id2label.get(int(corrected[i]), "O").endswith(f"-{entity_class}")
                        for i in match_tokens
                    )
                    if has_correct_entity:
                        self._apply_bio_sequence(
                            corrected, match_tokens, entity_class, label2id
                        )
                        self.stats[f"repaired_{entity_class}"] += len(match_tokens)
                        logger.debug(
                            f"Regex repaired {len(match_tokens)} tokens for {entity_class}: "
                            f"'{match.group()}'"
                        )

        return corrected

    @staticmethod
    def _apply_bio_sequence(
        corrected: np.ndarray,
        match_tokens: List[int],
        entity_class: str,
        label2id: Dict[str, int],
    ) -> None:
        """Assign a proper B-/I- sequence to the given token indices."""
        for i, tok_idx in enumerate(match_tokens):
            bio_prefix = "B-" if i == 0 else "I-"
            label_str = f"{bio_prefix}{entity_class}"
            label_id = label2id.get(label_str)
            if label_id is None:
                break  # entity class not in model vocabulary — skip
            corrected[tok_idx] = label_id

    def find_regex_hints(self, text: str) -> List[tuple]:
        """Return all validated regex matches in text as (start, end, entity_class) tuples."""
        hints = []
        for entity_class, compiled in self._compiled.items():
            config = self.patterns[entity_class]
            for match in compiled.finditer(text):
                if config.validator_fn and not config.validator_fn(match.group()):
                    continue
                hints.append((match.start(), match.end(), entity_class))
        return hints

    def validate_predictions(
        self,
        text: str,
        offset_mapping: np.ndarray,
        preds: np.ndarray,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        entities_to_validate: Optional[set] = None,
    ) -> np.ndarray:
        """Precision filter: demote entity predictions that don't match regex patterns.

        For each predicted entity span of a validatable type, check if the span
        text matches the corresponding regex pattern. If not, demote to O.

        Only applies to entities in ``entities_to_validate``.
        """
        if entities_to_validate is None:
            return preds

        corrected = preds.copy()
        o_id = label2id.get("O", 0)

        # Assemble predicted entity spans from preds
        spans: List[tuple] = []  # (entity_class, [token_indices])
        current_class = None
        current_indices = []

        for idx, pid in enumerate(corrected):
            label = id2label.get(int(pid), "O")
            if label.startswith("B-"):
                if current_class and current_indices:
                    spans.append((current_class, current_indices))
                current_class = label[2:]
                current_indices = [idx]
            elif label.startswith("I-") and current_class == label[2:]:
                current_indices.append(idx)
            else:
                if current_class and current_indices:
                    spans.append((current_class, current_indices))
                current_class = None
                current_indices = []

        if current_class and current_indices:
            spans.append((current_class, current_indices))

        # Validate each span against regex
        for entity_class, indices in spans:
            if entity_class not in entities_to_validate:
                continue

            compiled = self._compiled.get(entity_class)
            if compiled is None:
                continue

            # Get span character range from offset mapping
            starts = [int(offset_mapping[i][0]) for i in indices
                      if int(offset_mapping[i][0]) != int(offset_mapping[i][1])]
            ends = [int(offset_mapping[i][1]) for i in indices
                    if int(offset_mapping[i][0]) != int(offset_mapping[i][1])]

            if not starts or not ends:
                continue

            span_start = min(starts)
            span_end = max(ends)
            span_text = text[span_start:span_end]

            # Check if any regex match covers this span
            config = self.patterns[entity_class]
            match_found = False
            for match in compiled.finditer(text):
                if config.validator_fn and not config.validator_fn(match.group()):
                    continue
                # Match must overlap significantly with the span
                if match.start() <= span_start and match.end() >= span_end:
                    match_found = True
                    break
                if match.start() >= span_start and match.end() <= span_end:
                    match_found = True
                    break

            if not match_found:
                # Demote: entity prediction doesn't match regex pattern
                for i in indices:
                    corrected[i] = o_id
                self.stats[f"demoted_{entity_class}"] += len(indices)
                logger.debug(
                    f"Regex demoted {len(indices)} tokens of {entity_class}: "
                    f"'{span_text}'"
                )

        return corrected

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        self.stats.clear()
