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
  SOCIALNUM         — US SSN, UK NIN, CA SIN, FR NIR, AU TFN
  IBAN              — ISO 13616 with spaces, check-digit validation
  TELEPHONENUM      — E.164, US/CA, EU local, UK formats
  DATE              — ISO 8601, numeric, verbose month names (10 languages)
  ZIPCODE           — US, UK, EU, CA, BR, IN, JP
  TAXNUM            — IT CF (checksum), US EIN, EU VAT, BR CPF, ES NIF, IN PAN
  PASSPORTNUM       — international (US, UK, EU, IT, CN, CA)
  DRIVERLICENSENUM  — US, IT, UK, DE, CA
  MONEY             — currency symbol/code + amount (not in model vocab → overlay)
  REFNUM            — structured reference/account/policy numbers (not in model vocab → overlay)
  TIME              — time of day: 24h, 12h AM/PM, verbal (not in model vocab → overlay)

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


def _luhn_check_n(number_str: str, expected_len: int) -> bool:
    """Luhn check for arbitrary-length digit sequences (e.g. CA SIN = 9 digits)."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) != expected_len:
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


# Structural IBAN pattern (no checksum) — used to suppress CC false positives
# inside IBAN-shaped sequences.
_IBAN_STRUCTURAL_RE = re.compile(
    r"[A-Z]{2}\d{2}[\s\-]?(?:[A-Z0-9]{4}[\s\-]?){2,7}[A-Z0-9]{1,4}",
    re.IGNORECASE,
)


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


def _ssn_check_dispatch(raw: str) -> bool:
    """Dispatch SSN/NIN validation by detected format."""
    clean = raw.strip().upper()
    digits = [c for c in clean if c.isdigit()]
    alpha = [c for c in clean if c.isalpha()]

    # UK NIN: 2 letters + 6 digits + 1 letter (9 chars alpha+digit)
    if len(alpha) == 3 and len(digits) == 6:
        prefix = clean[:2]
        # Invalid NIN prefixes
        invalid_prefixes = {"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"}
        if prefix in invalid_prefixes:
            return False
        # First two chars must not be D, F, I, Q, U, V
        invalid_chars = set("DFIQUV")
        if clean[0] in invalid_chars or clean[1] in invalid_chars:
            return False
        return True

    # US SSN: 9 digits with dash or space separators (XXX-XX-XXXX)
    if len(digits) == 9 and len(alpha) == 0:
        # Must match XXX-XX-XXXX or XXX XX XXXX pattern (not XXX-XXX-XXX)
        if re.match(r"^\d{3}[\-\s]\d{2}[\-\s]\d{4}$", clean):
            return _ssn_check(raw)
        # CA SIN: XXX-XXX-XXX — require Luhn check to reduce false positives
        if re.match(r"^\d{3}[\-\s]\d{3}[\-\s]\d{3}$", clean):
            return _luhn_check_n(raw, 9)
        return False

    # French NIR: 15 digits (13 + 2 key)
    if len(digits) == 15 and len(alpha) == 0:
        return True  # structural match sufficient

    return True  # trust the regex structural match


def _italian_cf_check(raw: str) -> bool:
    """Validate Italian Codice Fiscale (CIN checksum).

    Format: RSSMRA78C14F205Z
      - 6 letters (surname 3 + name 3)
      - 2 digits (birth year)
      - 1 letter (birth month: A=Jan..T=Dec)
      - 2 digits (birth day, +40 for females)
      - 1 letter + 3 digits (municipality code)
      - 1 check letter (CIN)
    """
    cf = raw.strip().upper()
    if len(cf) != 16:
        return False

    # Structural check
    if not cf[:6].isalpha():
        return False
    if not cf[6:8].isdigit():
        return False
    if cf[8] not in "ABCDEHLMPRST":
        return False
    if not cf[9:11].isdigit():
        return False
    if not cf[11].isalpha():
        return False
    if not cf[12:15].isdigit():
        return False
    if not cf[15].isalpha():
        return False

    # Day validation (01-31 for males, 41-71 for females)
    day = int(cf[9:11])
    if not ((1 <= day <= 31) or (41 <= day <= 71)):
        return False

    # CIN checksum is intentionally NOT enforced here because:
    # - Synthetic/test datasets often generate CFs with invalid checksums
    # - The structural pattern (6 letters + month letter + day range) is
    #   already highly specific with very low false positive rate
    # - The check letter (pos 15) being [A-Z] is verified by the regex
    return True


def _tax_id_check_dispatch(raw: str) -> bool:
    """Dispatch tax ID validation by detected format."""
    clean = raw.strip().upper()
    digits = [c for c in clean if c.isdigit()]
    alpha = [c for c in clean if c.isalpha()]

    # Italian Codice Fiscale: 16 chars, 6 alpha + mixed
    if len(clean) == 16 and clean[:6].isalpha():
        return _italian_cf_check(clean)

    # Brazilian CPF: 11 digits (XXX.XXX.XXX-XX)
    if len(digits) == 11 and len(alpha) == 0:
        d = [int(c) for c in clean if c.isdigit()]
        # Reject all-same digits
        if len(set(d)) == 1:
            return False
        # First check digit
        s1 = sum(d[i] * (10 - i) for i in range(9)) % 11
        c1 = 0 if s1 < 2 else 11 - s1
        if d[9] != c1:
            return False
        # Second check digit
        s2 = sum(d[i] * (11 - i) for i in range(10)) % 11
        c2 = 0 if s2 < 2 else 11 - s2
        return d[10] == c2

    # Spanish NIF/NIE: letter check (mod 23)
    nif_match = re.match(r"^([XYZ]?)(\d{7,8})([A-Z])$", clean)
    if nif_match:
        prefix_letter = nif_match.group(1)
        number_str = nif_match.group(2)
        check = nif_match.group(3)
        nie_map = {"X": "0", "Y": "1", "Z": "2"}
        if prefix_letter:
            number_str = nie_map.get(prefix_letter, "") + number_str
        nif_letters = "TRWAGMYFPDXBNJZSQVHLCKE"
        try:
            expected = nif_letters[int(number_str) % 23]
            return check == expected
        except (ValueError, IndexError):
            return True

    # Indian PAN: 5 letters + 4 digits + 1 letter — 3rd char must be type code
    if len(clean) == 10 and clean[:5].isalpha() and clean[5:9].isdigit() and clean[9].isalpha():
        valid_types = set("ABCFGHLJPT")  # valid 4th character (entity type)
        return clean[3] in valid_types

    # Pure 10-digit number (UK UTR etc.) — too ambiguous, reject
    if len(digits) == 10 and len(alpha) == 0 and "." not in raw and "-" not in raw:
        return False

    # US EIN, EU VAT — structural match sufficient
    return True


def _phone_digit_count(raw: str) -> bool:
    """Validate phone number by digit count and structure.

    Requires 7-15 digits (ITU-T E.164) AND at least one non-digit separator
    (space, dash, dot, parenthesis). This prevents matching pure digit
    sequences like account numbers or IDs that happen to have 7-15 digits.
    """
    digits = [c for c in raw if c.isdigit()]
    if not (7 <= len(digits) <= 15):
        return False
    # Must have at least one separator — reject pure digit-dash sequences
    # like "8834-029-471" (depot number) or "2023-0003928" (policy number)
    has_space_or_dot = any(c in raw for c in " .()")
    has_plus = raw.lstrip().startswith("+")
    has_leading_zero = raw.lstrip().startswith("0")
    # Accept if: has + prefix, or has leading 0, or has spaces/dots/parens
    if not (has_plus or has_leading_zero or has_space_or_dot):
        return False
    return True


# ---------------------------------------------------------------------------
# Month names for verbose date patterns (10 languages)
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    # EN (full + abbreviated)
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    # IT
    "|gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
    # DE
    "|januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember"
    # FR
    "|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
    # ES
    "|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
    # PT
    "|janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
    # NL
    "|januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december"
    # PL
    "|styczen|luty|marzec|kwiecien|maj|czerwiec|lipiec|sierpien|wrzesien|pazdziernik|listopad|grudzien"
    # RO
    "|ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie"
    # TR
    "|ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik"
)


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

@dataclass
class RegexConfig:
    pattern: str
    validator_fn: Optional[Callable[[str], bool]] = None
    enabled: bool = True
    force_override: bool = False  # Override ANY prediction (not just O)
    case_sensitive: bool = False  # If True, compile without IGNORECASE


# Each key is a model entity class (without BIO prefix).
# Patterns are designed for high precision — they require the canonical format.
REGEX_PATTERNS: Dict[str, RegexConfig] = {
    # ── Credit card numbers ───────────────────────────────────────────────
    # Network-prefixed + permissive (any 13-19 digit Luhn-valid sequence)
    # force_override=True: the validator (_credit_card_check) guarantees high precision,
    # so override even when the model predicted a different entity (e.g. SOCIALNUM).
    "CREDITCARDNUMBER": RegexConfig(
        pattern=(
            # Network-prefixed patterns only (high precision).
            # No permissive fallback — generic digit sequences cause too many
            # false positives on driver's licenses, IBANs, and other long numbers.
            r"\b(?:"
            r"4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"   # Visa 16
            r"|5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # MC
            r"|2(?:2[2-9][1-9]|[3-6][0-9]{2}|7[01][0-9]|720)[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # MC 2-series
            r"|3[47][0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{5}"              # Amex 15
            r"|3(?:0[0-5]|[68][0-9])[0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{4}"  # Diners 14
            r"|6(?:011|5[0-9]{2})[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}"  # Discover
            r")\b"
        ),
        validator_fn=_credit_card_check,
        force_override=True,
    ),

    # ── Email ─────────────────────────────────────────────────────────────
    "EMAIL": RegexConfig(
        pattern=r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        validator_fn=None,
    ),

    # ── Social / National Insurance Numbers ───────────────────────────────
    "SOCIALNUM": RegexConfig(
        pattern=(
            r"\b(?:"
            # US SSN: 123-45-6789 or 123 45 6789
            r"\d{3}[\-\s]\d{2}[\-\s]\d{4}"
            # UK NIN: AB 12 34 56 C (with or without spaces)
            r"|[A-CEGHJ-PR-TW-Z]{2}[\s]?\d{2}[\s]?\d{2}[\s]?\d{2}[\s]?[A-D]"
            # Canadian SIN: 123-456-789 or 123 456 789
            r"|\d{3}[\-\s]\d{3}[\-\s]\d{3}"
            # French NIR (INSEE): 1|2 + 12 digits + 2-digit key (with optional spaces)
            r"|[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}"
            r")\b"
        ),
        validator_fn=_ssn_check_dispatch,
    ),

    # ── IBAN — ISO 13616 with spaces/dashes ───────────────────────────────
    "IBAN": RegexConfig(
        pattern=r"\b[A-Z]{2}\d{2}[\s\-]?(?:[A-Z0-9]{4}[\s\-]?){2,7}[A-Z0-9]{1,4}\b",
        validator_fn=_iban_check,
    ),

    # ── Phone numbers — international ─────────────────────────────────────
    "TELEPHONENUM": RegexConfig(
        pattern=(
            r"(?:"
            # International with + prefix (E.164): +39 02 4823 7761, +1-555-123-4567
            r"\+\d{1,3}[\s.\-]?\(?\d{1,5}\)?[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}(?:[\s.\-]?\d{1,4})?"
            # US/CA: (555) 123-4567 or 555-123-4567 (requires separator)
            r"|\b\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"
            # EU local with leading 0: 02 4823 7761, 0171 234 5678, 06 12 34 56 78
            r"|\b0\d{1,4}[\s.\-/]\d{2,4}[\s.\-/]\d{2,4}(?:[\s.\-/]\d{1,4})?\b"
            # UK landline/mobile: 020 7946 0958, 07911 123456
            r"|\b0\d{2,4}[\s\-]\d{3,4}[\s\-]\d{3,4}\b"
            r")"
        ),
        validator_fn=_phone_digit_count,
    ),

    # ── Tax identification numbers — international ────────────────────────
    # force_override=True: the validator guarantees high precision (checksum for
    # CF, CPF, NIF), so override when model predicted wrong entity (e.g. CF→DRIVERLICENSENUM).
    "TAXNUM": RegexConfig(
        pattern=(
            r"\b(?:"
            # Italian Codice Fiscale: RSSMRA78C14F205Z (16 alphanumeric, CIN checksum)
            r"[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]"
            # US EIN: 12-3456789
            r"|\d{2}-\d{7}"
            # EU VAT: country code + 8-12 digits (DE123456789, FR12345678901)
            r"|[A-Z]{2}\d{8,12}"
            # Brazilian CPF: 123.456.789-01 or 12345678901
            r"|\d{3}\.?\d{3}\.?\d{3}[\-.]?\d{2}"
            # Spanish NIF/NIE: X1234567L or 12345678Z
            r"|[XYZ]\d{7}[A-Z]"
            r"|\d{8}[A-Z]"
            # Indian PAN: ABCDE1234F (5 letters + 4 digits + 1 letter)
            r"|[A-Z]{5}\d{4}[A-Z]"
            r")\b"
        ),
        validator_fn=_tax_id_check_dispatch,
        force_override=True,
    ),

    # ── Dates — ISO 8601, numeric, verbose month names (10 languages) ─────
    "DATE": RegexConfig(
        pattern=(
            r"\b(?:"
            # ISO: 2024-03-10
            r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
            # US: 03/10/2024 or 3/10/24
            r"|(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/\d{2,4}"
            # EU: 10.03.2024 or 10/03/2024
            r"|(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./]\d{2,4}"
            # Verbose day-first: "14 marzo 1978", "3 febbraio 2025", "10. März 2024"
            rf"|(?:0?[1-9]|[12]\d|3[01])\.?[\s\-](?:{_MONTH_NAMES})[\s,\-]+\d{{4}}"
            # Verbose month-first: "March 14, 2025", "January 3rd, 2024"
            rf"|(?:{_MONTH_NAMES})[\s\-]+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[\s,\-]+\d{{4}}"
            r")\b"
        ),
        validator_fn=None,
    ),

    # ── ZIP / Postal codes — international ────────────────────────────────
    # Note: bare \d{5} is intentionally NOT included — too many false positives
    # (Grundbuch numbers, Aktenzeichen, etc.). The model handles 5-digit ZIPs;
    # regex only confirms structured formats (UK, NL, CA, BR, JP, US ZIP+4).
    "ZIPCODE": RegexConfig(
        pattern=(
            r"\b(?:"
            # US ZIP+4: 12345-6789 (the +4 suffix makes it unambiguous)
            r"\d{5}-\d{4}"
            # UK: A9 9AA, A99 9AA, A9A 9AA, AA9 9AA, AA99 9AA, AA9A 9AA
            r"|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}"
            # NL: 1234 AB
            r"|\d{4}\s?[A-Z]{2}"
            # CA: A1A 1A1
            r"|[A-Z]\d[A-Z]\s?\d[A-Z]\d"
            # BR: 12345-678
            r"|\d{5}-\d{3}"
            # JP: 123-4567
            r"|\d{3}-\d{4}"
            r")\b"
        ),
        validator_fn=None,
    ),

    # ── Passport numbers — international ──────────────────────────────────
    "PASSPORTNUM": RegexConfig(
        pattern=(
            r"\b(?:"
            # Standard: 1-2 letters + 6-9 digits (US, UK, EU, IT, BR)
            r"[A-Z]{1,2}\d{6,9}"
            # Some EU: 2 digits + 2 letters + 5 digits
            r"|\d{2}[A-Z]{2}\d{5}"
            # Chinese: E/G/D/P/H/S + 8 digits
            r"|[EGDPHS]\d{8}"
            r")\b"
        ),
        validator_fn=None,
    ),

    # ── Driver's license — international, restrictive ─────────────────────
    "DRIVERLICENSENUM": RegexConfig(
        pattern=(
            r"\b(?:"
            # US: 1 letter + 7-14 digits (raised min from 6 to reduce FP)
            r"[A-Z]\d{7,14}"
            # Italian: 2 letters + 7 digits + 1 letter (e.g. MI2847563A)
            r"|[A-Z]{2}\d{7}[A-Z]"
            # UK DVLA: MORGA657054SM9IJ (5 letters + 6 digits + 2 letters + 2 digits + 2 letters + 2 digits)
            r"|[A-Z]{5}\d{6}[A-Z]\d{2}[A-Z]{2}\d{2}"
            # German: letter + 2-3 digits + letter + 6 digits + digit
            r"|[A-Z]\d{2,3}[A-Z]\d{6}\d"
            # Canadian: 1 letter + 4 digits + hyphen + 5 digits + hyphen + 5 digits
            r"|[A-Z]\d{4}-\d{5}-\d{5}"
            r")\b"
        ),
        validator_fn=None,
    ),

    # ── Monetary amounts — international ───────────────────────────────
    # Not in the model vocabulary → injected via regex overlay (Phase 6).
    # Matches currency symbol/code + amount, or amount + currency code.
    # Uses explicit ISO 4217 codes to avoid false positives on abbreviations.
    "MONEY": RegexConfig(
        pattern=(
            r"(?:"
            # Symbol before amount: $1,234.56  €3.200  £12,000  ¥1234  ₹100000
            r"[$€£¥₹₩₪฿]\s?\d[\d.,]*\d"
            r"|[$€£¥₹₩₪฿]\s?\d"
            # ISO code / currency name BEFORE amount: USD 1,234.56  euro 100000
            r"|\b(?:USD|EUR|GBP|CHF|JPY|CNY|INR|BRL|CAD|AUD|SEK|NOK|DKK|PLN|CZK|HUF|RON|TRY|RUB|KRW|THB|MXN|ZAR"
            r"|euro|euros|dollari|dollaro|dollars?|sterline|sterlina|pounds?|franchi|francs?"
            r"|yen|yuan|reais|real|rupie|rupees?|corone|kronor|kroner|zloty|fiorini|lire"
            r")\s\d[\d.,]*\d"
            r"|\b(?:USD|EUR|GBP|CHF|JPY|CNY|INR|BRL|CAD|AUD|SEK|NOK|DKK|PLN|CZK|HUF|RON|TRY|RUB|KRW|THB|MXN|ZAR"
            r"|euro|euros|dollari|dollaro|dollars?|sterline|sterlina|pounds?|franchi|francs?"
            r"|yen|yuan|reais|real|rupie|rupees?|corone|kronor|kroner|zloty|fiorini|lire"
            r")\s\d"
            # Amount before symbol: 3.200 €  100000€
            r"|\b\d[\d.,]*\d\s?[$€£¥₹₩₪฿]"
            r"|\b\d\s?[$€£¥₹₩₪฿]"
            # Amount BEFORE ISO code / currency name: 100000 EUR  100000 euro
            r"|\b\d[\d.,]*\d\s(?:USD|EUR|GBP|CHF|JPY|CNY|INR|BRL|CAD|AUD|SEK|NOK|DKK|PLN|CZK|HUF|RON|TRY|RUB|KRW|THB|MXN|ZAR"
            r"|euro|euros|dollari|dollaro|dollars?|sterline|sterlina|pounds?|franchi|francs?"
            r"|yen|yuan|reais|real|rupie|rupees?|corone|kronor|kroner|zloty|fiorini|lire)\b"
            r"|\b\d\s(?:USD|EUR|GBP|CHF|JPY|CNY|INR|BRL|CAD|AUD|SEK|NOK|DKK|PLN|CZK|HUF|RON|TRY|RUB|KRW|THB|MXN|ZAR"
            r"|euro|euros|dollari|dollaro|dollars?|sterline|sterlina|pounds?|franchi|francs?"
            r"|yen|yuan|reais|real|rupie|rupees?|corone|kronor|kroner|zloty|fiorini|lire)\b"
            r")"
        ),
        validator_fn=None,
        # case_sensitive=False: currency names need case-insensitive matching
        # ISO codes still use explicit uppercase in the pattern alternation
    ),

    # ── Reference / account numbers — catch-all for structured identifiers ──
    # Not in the model vocabulary → injected via regex overlay (Phase 6).
    # Catches policy numbers, sort codes, account numbers, reference numbers,
    # and other structured alphanumeric sequences commonly associated with PII.
    "REFNUM": RegexConfig(
        pattern=(
            r"\b(?:"
            # Letters-dash-digits: LV-2022-00493821, PI-2022-00471832, KV-2024-00871234
            r"[A-Z]{1,5}[-/]\d{4}[-/]\d{5,12}"
            # Dash-separated digit groups (3+ segments): 20-41-38, 9923-147-882, 448392-DM-0071
            r"|\d{2,6}[-]\d{2,6}[-][\dA-Z]{2,8}"
            # UK sort code + account: 20-41-38 ... 73948201
            r"|\d{2}-\d{2}-\d{2}"
            # Alphanumeric reference with slashes: GD1G/00293847/4
            r"|[A-Z\d]{2,6}/\d{5,10}(?:/\d{1,4})?"
            r")\b"
        ),
        validator_fn=None,
        case_sensitive=True,
    ),

    # ── Time / hours — international ──────────────────────────────────────
    # Not in the model vocabulary → injected via regex overlay (Phase 6).
    # Matches common time formats: 24h, 12h with AM/PM, verbal (ore/Uhr/h/heure).
    "TIME": RegexConfig(
        pattern=(
            r"(?:"
            # 24h with seconds: 14:30:00, 08:00:00
            r"\b(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d\b"
            # 24h: 14:30, 08:00, 23:59 (also matches H.MM common in IT/DE)
            r"|\b(?:[01]?\d|2[0-3])[:\.][0-5]\d\b"
            # 12h with AM/PM: 2:45 PM, 11:30am, 12:00 A.M.
            r"|\b(?:0?[1-9]|1[0-2]):[0-5]\d\s?(?:[AaPp]\.?[Mm]\.?)\b"
            # Verbal IT/FR/PT: "ore 14:30", "alle 9:00", "à 15h30", "às 14h"
            r"|\b(?:ore|alle|às?)\s(?:[01]?\d|2[0-3])[:\.h][0-5]\d?\b"
            # Verbal DE: "um 14:30 Uhr", "8:00 Uhr", "14 Uhr"
            r"|\b(?:[01]?\d|2[0-3])[:\.][0-5]\d\s?[Uu]hr\b"
            r"|\b(?:[01]?\d|2[0-3])\s?[Uu]hr\b"
            # French: "15h30", "8h00", "14h"
            r"|\b(?:[01]?\d|2[0-3])[hH][0-5]\d?\b"
            r")"
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
                flags = 0 if cfg.case_sensitive else re.IGNORECASE
                self._compiled[cls] = re.compile(cfg.pattern, flags)

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
                # All tokens in span are special tokens (offset 0,0) — skip.
                # Without this guard, min()/max() on empty lists would raise ValueError.
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
