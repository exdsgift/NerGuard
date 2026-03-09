"""
General NER prompt provider for NerGuard — WikiNeural compatibility experiment.

Uses a generic prompt that does not assume any PII context.
Demonstrates that the routing principle works with standard NER labels
(PER, ORG, LOC, MISC) without domain-specific engineering.
"""

import json
from typing import Dict, Optional, Set

from src.core.base_prompt import PromptProvider
from src.tasks.wikiner.config import WIKINER_ENTITY_CLASSES_WITH_O


WIKINER_SPAN_PROMPT = """You are a named entity recognition expert. Classify the highlighted span.

CONTEXT:
{context}

TARGET SPAN: "{span_text}"
TOKEN COUNT: {token_count}
MODEL PREDICTION: {entity_class}

Is this span a {entity_class}? Return the entity class if correct, a different class if misclassified, or "O" if not an entity.

## ENTITY TYPES:
- PER: person names (e.g. "Barack Obama", "Marie Curie")
- ORG: organizations, companies, institutions (e.g. "Google", "United Nations", "Harvard")
- LOC: locations, countries, cities, geographical features (e.g. "Paris", "Amazon River")
- MISC: miscellaneous named entities (e.g. nationalities, events, languages, works of art)
- O: not a named entity

## EXAMPLES:

Example 1 (confirm):
SPAN: "Angela Merkel" | MODEL: PER
Answer: {{"reasoning": "Political leader, confirmed person name", "entity_class": "PER"}}

Example 2 (reject):
SPAN: "the" | MODEL: ORG
Answer: {{"reasoning": "Common word, not an organization", "entity_class": "O"}}

Example 3 (correct type):
SPAN: "Berlin" | MODEL: ORG
Answer: {{"reasoning": "Capital city of Germany, this is a location not an organization", "entity_class": "LOC"}}

Valid entity types (or "O"):
{entity_classes_str}

Respond with JSON only:
{{"reasoning": "<10-15 words>", "entity_class": "<CLASS_NAME>"}}"""


WIKINER_O_SPAN_PROMPT = """You are a named entity recognition expert. The model tagged this span as non-entity (O), but it is uncertain.

CONTEXT:
{context}

TARGET SPAN: "{span_text}"
TOKEN COUNT: {token_count}

## TASK:
Is this span a named entity that the model missed?

- PER: person names
- ORG: organizations, companies, institutions
- LOC: locations, geographical features
- MISC: nationalities, events, languages, miscellaneous proper nouns

If it is a named entity, return its class. If not, return "O".

Valid types: {target_labels_str}

Respond with JSON only:
{{"is_pii": true/false, "reasoning": "<10-15 words>", "entity_class": "<CLASS_NAME>"}}"""


class WikinerPromptProvider(PromptProvider):
    """Generic NER prompts for PER/ORG/LOC/MISC disambiguation."""

    def __init__(self):
        self._entity_classes_str = json.dumps(sorted(WIKINER_ENTITY_CLASSES_WITH_O), indent=2)

    def system_message(self) -> str:
        return "You are a named entity recognition expert. Output only valid JSON."

    def span_prompt(self, context, span_text, token_count, entity_class, entity_classes_str):
        return WIKINER_SPAN_PROMPT.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            entity_class=entity_class,
            entity_classes_str=entity_classes_str,
        )

    def o_span_prompt(self, context, span_text, token_count, target_labels_str):
        return WIKINER_O_SPAN_PROMPT.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            target_labels_str=target_labels_str,
        )

    def valid_entity_classes(self) -> Set[str]:
        return WIKINER_ENTITY_CLASSES_WITH_O

    def entity_class_aliases(self) -> Dict[str, str]:
        return {
            "person": "PER",
            "people": "PER",
            "organization": "ORG",
            "company": "ORG",
            "location": "LOC",
            "place": "LOC",
            "country": "LOC",
            "city": "LOC",
            "miscellaneous": "MISC",
        }

    def entity_classes_str(self) -> str:
        return self._entity_classes_str
