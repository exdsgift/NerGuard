"""
PII-specific prompt provider for NerGuard.

Wraps the existing prompt templates (V9, V13, V14_SPAN, etc.) into the
generic PromptProvider interface. The key PII-specific elements are:
- System message: "You are a PII classification expert"
- Recall bias: "When uncertain, prefer classifying as PII over O"
- NVIDIA alias mapping for cross-dataset evaluation
"""

import json
from typing import Dict, Optional, Set

from src.core.base_prompt import PromptProvider
from src.tasks.pii.config import PII_ENTITY_CLASSES_WITH_O


# NVIDIA dataset alias -> base model entity class
NVIDIA_CLASS_TO_BASE = {
    "first_name": "GIVENNAME",
    "last_name": "SURNAME",
    "middle_name": "GIVENNAME",
    "phone_number": "TELEPHONENUM",
    "cell_phone": "TELEPHONENUM",
    "fax_number": "TELEPHONENUM",
    "street_address": "STREET",
    "zipcode": "ZIPCODE",
    "postcode": "ZIPCODE",
    "ssn": "SOCIALNUM",
    "social_security_number": "SOCIALNUM",
    "tax_id": "TAXNUM",
    "driver_license": "DRIVERLICENSENUM",
    "drivers_license": "DRIVERLICENSENUM",
    "certificate_license_number": "DRIVERLICENSENUM",
    "national_id": "IDCARDNUM",
    "passport_number": "PASSPORTNUM",
    "credit_debit_card": "CREDITCARDNUMBER",
    "date_of_birth": "DATE",
    "sexuality": "GENDER",
}

# Extended entity classes: base 20 + NVIDIA aliases
EXTENDED_PII_CLASSES_WITH_O = PII_ENTITY_CLASSES_WITH_O | set(NVIDIA_CLASS_TO_BASE.keys())


class PIIPromptProvider(PromptProvider):
    """PII-specific prompts for LLM disambiguation.

    Args:
        span_prompt_version: Which span prompt to use (V14_SPAN, V15_SPAN, V16_SPAN).
        use_extended_labels: Accept NVIDIA-alias entity names in LLM responses.
    """

    def __init__(
        self,
        span_prompt_version: str = "V14_SPAN",
        use_extended_labels: bool = False,
    ):
        from src.inference.prompts import PROMPTS, PROMPT_V14_SPAN, PROMPT_O_SPAN

        self._span_template = PROMPTS.get(span_prompt_version, PROMPT_V14_SPAN)
        self._o_span_template = PROMPT_O_SPAN
        self._use_extended = use_extended_labels or (span_prompt_version == "V16_SPAN")
        self._entity_classes_str = json.dumps(sorted(PII_ENTITY_CLASSES_WITH_O), indent=2)

    def system_message(self) -> str:
        return "You are a PII classification expert. Output only valid JSON."

    def span_prompt(self, context, span_text, token_count, entity_class, entity_classes_str):
        return self._span_template.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            entity_class=entity_class,
            entity_classes_str=entity_classes_str,
        )

    def o_span_prompt(self, context, span_text, token_count, target_labels_str):
        return self._o_span_template.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            target_labels_str=target_labels_str,
        )

    def valid_entity_classes(self) -> Set[str]:
        if self._use_extended:
            return EXTENDED_PII_CLASSES_WITH_O
        return PII_ENTITY_CLASSES_WITH_O

    def entity_class_aliases(self) -> Dict[str, str]:
        return NVIDIA_CLASS_TO_BASE

    def entity_classes_str(self) -> str:
        """JSON string of valid entity classes for prompt formatting."""
        return self._entity_classes_str
