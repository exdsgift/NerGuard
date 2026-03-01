"""
Prompt templates for LLM disambiguation in NerGuard.

This module contains the prompt templates used for LLM-based
disambiguation of uncertain NER predictions. V9 is the production
default (best performance with GPT-4o: +70 net improvement). V12
adds enhanced BIO rules with explicit few-shot examples.
"""

import json
from src.core.constants import VALID_LABELS

VALID_LABELS_STR = json.dumps(VALID_LABELS, indent=2)


PROMPT_V9 = """The model detected potential PII in the marked token >>> <<<. Your task is to classify it.

{context}

PREVIOUS TAG: {prev_label}

IMPORTANT: When uncertain, prefer classifying as PII over O. Missing PII is worse than false positive.

PII TYPES:
- Names: GIVENNAME, SURNAME, TITLE
- IDs: CREDITCARDNUMBER, TELEPHONENUM, EMAIL, SOCIALNUM, PASSPORTNUM
- Location: STREET, CITY, BUILDINGNUM, ZIPCODE
- Time: DATE, TIME, AGE
- Other: SEX, GENDER, USERNAME, IBAN

BIO RULES:
- B-X: First token of entity type X
- I-X: Continuation of entity (if previous is B-X or I-X)
- O: ONLY for clearly non-PII (articles, prepositions, punctuation)

{valid_labels_str}

{{"label": "TAG"}}"""


PROMPT_V12 = """You are a PII detection expert. Classify the marked token >>> <<< with the correct BIO label.

CONTEXT:
{context}

PREVIOUS TOKEN LABEL: {prev_label}
MODEL PREDICTION: {current_pred}

## CRITICAL BIO RULES (NEVER VIOLATE):

1. **B- (Begin)**: ONLY for the FIRST token of an entity span
2. **I- (Inside)**: ONLY for CONTINUATION tokens within the SAME entity
3. **I- can NEVER follow O** - This is a hard constraint
4. **I- can NEVER follow a DIFFERENT entity type** - I-X must follow B-X or I-X

## DECISION TREE:

1. Is previous label O?
   → You can ONLY output B-X or O (NEVER I-X)

2. Is previous label B-X or I-X?
   → If this token CONTINUES the same entity → I-X
   → If this token is a NEW entity → B-Y
   → If this token is not PII → O

3. Is this clearly not PII (article, preposition, punctuation)?
   → O

## FEW-SHOT EXAMPLES:

Example 1 - Phone continuation:
Context: "Call >>> -1234 <<< for info"
Previous: I-TELEPHONENUM
Answer: {{"reasoning": "Continues phone number sequence", "label": "I-TELEPHONENUM"}}

Example 2 - Name boundary (two separate names):
Context: "Dear >>> Smith <<<, your order"
Previous: B-GIVENNAME (for "John")
Answer: {{"reasoning": "Smith is surname, new entity after given name", "label": "B-SURNAME"}}

Example 3 - Cannot use I- after O:
Context: "The >>> 555 <<< number is"
Previous: O
Answer: {{"reasoning": "First digit of phone, must start with B-", "label": "B-TELEPHONENUM"}}

Example 4 - Non-PII token:
Context: "Contact >>> the <<< office at"
Previous: O
Answer: {{"reasoning": "Article 'the' is not PII", "label": "O"}}

Example 5 - Credit card continuation:
Context: "Card: 4111 >>> 1111 <<< 1111 1111"
Previous: I-CREDITCARDNUMBER
Answer: {{"reasoning": "Continues credit card number sequence", "label": "I-CREDITCARDNUMBER"}}

VALID LABELS:
{valid_labels_str}

Respond with JSON only:
{{"reasoning": "<your analysis in 10-15 words>", "label": "<BIO_TAG>"}}"""


# Default prompt: V9 achieved best performance with GPT-4o (+70 net improvement)
PROMPT = PROMPT_V9

# Prompt registry for ablation experiments
PROMPTS = {
    "V9": PROMPT_V9,
    "V12": PROMPT_V12,
}
