"""
Biomedical NER prompt provider for NerGuard.

Provides LLM prompts for disambiguating Chemical and Disease entities
in biomedical text (PubMed abstracts). Unlike PII prompts, the bias
here is toward recall of disease/chemical mentions.
"""

import json
from typing import Dict, Optional, Set

from src.core.base_prompt import PromptProvider
from src.tasks.biomedical.config import BIO_ENTITY_CLASSES_WITH_O


BIOMEDICAL_SPAN_PROMPT = """You are a biomedical named entity recognition expert. Classify the highlighted span.

CONTEXT:
{context}

TARGET SPAN: "{span_text}"
TOKEN COUNT: {token_count}
MODEL PREDICTION: {entity_class}

Is this span a {entity_class}? Return the entity class if correct, a different class if misclassified, or "O" if not an entity.

## ENTITY TYPES:
- Chemical: drugs, compounds, chemical substances (e.g. "aspirin", "methotrexate", "NaCl")
- Disease: diseases, conditions, symptoms, disorders (e.g. "diabetes", "hepatotoxicity", "fever")
- O: not a biomedical entity

## EXAMPLES:

Example 1 (confirm):
SPAN: "methotrexate" | MODEL: Chemical
Answer: {{"reasoning": "Chemotherapy drug, confirmed chemical entity", "entity_class": "Chemical"}}

Example 2 (reject):
SPAN: "patients" | MODEL: Disease
Answer: {{"reasoning": "General term, not a disease name", "entity_class": "O"}}

Example 3 (correct type):
SPAN: "hepatotoxicity" | MODEL: Chemical
Answer: {{"reasoning": "Liver damage condition, this is a disease not a chemical", "entity_class": "Disease"}}

Valid entity types (or "O"):
{entity_classes_str}

Respond with JSON only:
{{"reasoning": "<10-15 words>", "entity_class": "<CLASS_NAME>"}}"""


BIOMEDICAL_O_SPAN_PROMPT = """You are a biomedical NER expert. The model tagged this span as non-entity (O), but it is uncertain.

CONTEXT:
{context}

TARGET SPAN: "{span_text}"
TOKEN COUNT: {token_count}

## TASK:
Is this span a Chemical or Disease entity that the model missed?

- Chemical: drugs, compounds, chemical substances, elements
- Disease: diseases, conditions, symptoms, disorders, syndromes

If it is a biomedical entity, return its class. If not, return "O".

Valid types: {target_labels_str}

Respond with JSON only:
{{"is_pii": true/false, "reasoning": "<10-15 words>", "entity_class": "<CLASS_NAME>"}}"""


class BiomedicalPromptProvider(PromptProvider):
    """Biomedical NER prompts for Chemical/Disease disambiguation."""

    def __init__(self):
        self._entity_classes_str = json.dumps(sorted(BIO_ENTITY_CLASSES_WITH_O), indent=2)

    def system_message(self) -> str:
        return "You are a biomedical named entity recognition expert. Output only valid JSON."

    def span_prompt(self, context, span_text, token_count, entity_class, entity_classes_str):
        return BIOMEDICAL_SPAN_PROMPT.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            entity_class=entity_class,
            entity_classes_str=entity_classes_str,
        )

    def o_span_prompt(self, context, span_text, token_count, target_labels_str):
        return BIOMEDICAL_O_SPAN_PROMPT.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            target_labels_str=target_labels_str,
        )

    def valid_entity_classes(self) -> Set[str]:
        return BIO_ENTITY_CLASSES_WITH_O

    def entity_class_aliases(self) -> Dict[str, str]:
        # Common aliases in biomedical literature
        return {
            "drug": "Chemical",
            "compound": "Chemical",
            "medication": "Chemical",
            "disorder": "Disease",
            "condition": "Disease",
            "symptom": "Disease",
            "syndrome": "Disease",
        }

    def entity_classes_str(self) -> str:
        return self._entity_classes_str
