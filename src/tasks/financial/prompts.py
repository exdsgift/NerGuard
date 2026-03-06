"""
Financial NER prompt provider for NerGuard.

Provides LLM prompts for disambiguating business entities in SEC filings
(M&A documents). 6 entity classes: parties to the deal, advisors, and revenues.
"""

import json
from typing import Dict, Optional, Set

from src.core.base_prompt import PromptProvider
from src.tasks.financial.config import BUSTER_ENTITY_CLASSES_WITH_O, BUSTER_SHORT_NAMES


FINANCIAL_SPAN_PROMPT = """You are a financial named entity recognition expert specializing in M&A (mergers and acquisitions) documents from SEC filings.

CONTEXT:
{context}

TARGET SPAN: "{span_text}"
TOKEN COUNT: {token_count}
MODEL PREDICTION: {entity_class}

Is this span correctly classified as {entity_class}? Return the entity class if correct, a different class if misclassified, or "O" if not an entity.

## ENTITY TYPES:
- Parties.BUYING_COMPANY: the company making the acquisition (buyer/acquirer)
- Parties.SELLING_COMPANY: the company selling assets or being divested from
- Parties.ACQUIRED_COMPANY: the company being acquired (target of the deal)
- Advisors.LEGAL_CONSULTING_COMPANY: law firms advising on the transaction
- Advisors.GENERIC_CONSULTING_COMPANY: investment banks, consultants, other advisors
- Generic_Info.ANNUAL_REVENUES: monetary amounts representing annual revenue figures
- O: not a relevant entity

## EXAMPLES:

Example 1 (confirm buyer):
SPAN: "Microsoft Corporation" | MODEL: Parties.BUYING_COMPANY
Answer: {{"reasoning": "Microsoft is acquiring the target company", "entity_class": "Parties.BUYING_COMPANY"}}

Example 2 (correct type):
SPAN: "Goldman Sachs" | MODEL: Parties.SELLING_COMPANY
Answer: {{"reasoning": "Goldman Sachs is a financial advisor, not a selling party", "entity_class": "Advisors.GENERIC_CONSULTING_COMPANY"}}

Example 3 (reject):
SPAN: "Delaware" | MODEL: Parties.ACQUIRED_COMPANY
Answer: {{"reasoning": "State name, not a company being acquired", "entity_class": "O"}}

Valid entity types (or "O"):
{entity_classes_str}

Respond with JSON only:
{{"reasoning": "<10-15 words>", "entity_class": "<CLASS_NAME>"}}"""


FINANCIAL_O_SPAN_PROMPT = """You are a financial NER expert on M&A documents. The model tagged this span as non-entity (O), but it is uncertain.

CONTEXT:
{context}

TARGET SPAN: "{span_text}"
TOKEN COUNT: {token_count}

## TASK:
Is this span a business entity that the model missed? Consider:
- Parties.BUYING_COMPANY: the acquirer
- Parties.SELLING_COMPANY: the seller/divestor
- Parties.ACQUIRED_COMPANY: the target being acquired
- Advisors.LEGAL_CONSULTING_COMPANY: law firms
- Advisors.GENERIC_CONSULTING_COMPANY: investment banks, consultants
- Generic_Info.ANNUAL_REVENUES: annual revenue figures

If it is a financial entity, return its class. If not, return "O".

Valid types: {target_labels_str}

Respond with JSON only:
{{"is_pii": true/false, "reasoning": "<10-15 words>", "entity_class": "<CLASS_NAME>"}}"""


class FinancialPromptProvider(PromptProvider):
    """Financial NER prompts for M&A entity disambiguation."""

    def __init__(self):
        self._entity_classes_str = json.dumps(sorted(BUSTER_ENTITY_CLASSES_WITH_O), indent=2)

    def system_message(self) -> str:
        return "You are a financial named entity recognition expert specializing in SEC filings and M&A transactions. Output only valid JSON."

    def span_prompt(self, context, span_text, token_count, entity_class, entity_classes_str):
        return FINANCIAL_SPAN_PROMPT.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            entity_class=entity_class,
            entity_classes_str=entity_classes_str,
        )

    def o_span_prompt(self, context, span_text, token_count, target_labels_str):
        return FINANCIAL_O_SPAN_PROMPT.format(
            context=context,
            span_text=span_text,
            token_count=token_count,
            target_labels_str=target_labels_str,
        )

    def valid_entity_classes(self) -> Set[str]:
        return BUSTER_ENTITY_CLASSES_WITH_O

    def entity_class_aliases(self) -> Dict[str, str]:
        return {
            "buyer": "Parties.BUYING_COMPANY",
            "acquirer": "Parties.BUYING_COMPANY",
            "buying company": "Parties.BUYING_COMPANY",
            "seller": "Parties.SELLING_COMPANY",
            "selling company": "Parties.SELLING_COMPANY",
            "target": "Parties.ACQUIRED_COMPANY",
            "acquired company": "Parties.ACQUIRED_COMPANY",
            "law firm": "Advisors.LEGAL_CONSULTING_COMPANY",
            "legal advisor": "Advisors.LEGAL_CONSULTING_COMPANY",
            "investment bank": "Advisors.GENERIC_CONSULTING_COMPANY",
            "advisor": "Advisors.GENERIC_CONSULTING_COMPANY",
            "consultant": "Advisors.GENERIC_CONSULTING_COMPANY",
            "revenue": "Generic_Info.ANNUAL_REVENUES",
            "revenues": "Generic_Info.ANNUAL_REVENUES",
        }

    def entity_classes_str(self) -> str:
        return self._entity_classes_str
