"""
Prompt templates for LLM disambiguation in NerGuard.

This module contains versioned prompt templates used for LLM-based
disambiguation of uncertain NER predictions.
"""

import json
from src.core.constants import VALID_LABELS

# JSON string of valid labels for prompts
VALID_LABELS_STR = json.dumps(VALID_LABELS, indent=2)


PROMPT_V11 = """You are a PII detection expert. A machine learning model flagged the token marked with >>> <<< as potentially containing personally identifiable information (PII). Your task is to determine the correct BIO label.

CONTEXT:
{context}

PREVIOUS TOKEN LABEL: {prev_label}

STEP 1 - ANALYZE THE TOKEN:
Think about what the marked token represents. Is it:
- A person's name (first name, last name, title)?
- An identifier (credit card, phone, email, SSN, passport)?
- A location (street, city, building number, zip code)?
- A time reference (date, time, age)?
- A common word (article, preposition, conjunction)?

STEP 2 - CHECK BIO CONSISTENCY:
- If previous label is B-X or I-X and this token CONTINUES the same entity → use I-X
- If this token STARTS a new entity → use B-X
- If this is clearly NOT PII → use O

STEP 3 - APPLY THE CONSERVATIVE RULE:
When uncertain between PII and O, ALWAYS choose PII. Missing real PII is worse than a false positive.

Use O ONLY for: articles (the, a, an), prepositions (of, in, at, for, with), conjunctions (and, or, but), standalone punctuation.

VALID LABELS:
{valid_labels_str}

Respond with JSON only:
{{"reasoning": "<your analysis in 10-15 words>", "label": "<BIO_TAG>"}}

EXAMPLES:
{{"reasoning": "Capitalized word after 'Dear' indicates given name", "label": "B-GIVENNAME"}}
{{"reasoning": "Continues phone number from previous B-TELEPHONENUM", "label": "I-TELEPHONENUM"}}
{{"reasoning": "Common preposition 'of' is not PII", "label": "O"}}"""


PROMPT_V10 = """TASK: The model flagged >>> <<< as potential PII. Determine the PII type.

{context}

PREVIOUS: {prev_label}

DEFAULT TO PII. Use O only for: articles (the, a, an), prepositions (of, in, at, for), conjunctions (and, or, but), punctuation alone.

TYPES:
GIVENNAME (John, Maria), SURNAME (Smith, Rossi), TITLE (Dr, Mr, Mrs)
CREDITCARDNUMBER (4111...), TELEPHONENUM (+39..., 555-1234), EMAIL (@domain)
SOCIALNUM (SSN), PASSPORTNUM, IDCARDNUM, DRIVERLICENSENUM, TAXNUM
STREET (Via Roma, 123 Main St), CITY (Rome, NYC), BUILDINGNUM, ZIPCODE
DATE (01/01/2020), TIME (14:30), AGE (25 years old)
SEX/GENDER (male, female), USERNAME (@user), IBAN

IF previous is B-X/I-X and token continues the entity → I-X
IF token starts new entity → B-X
IF CLEARLY an article/preposition/conjunction → O

{valid_labels_str}

{{"label": "TAG"}}"""


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


PROMPT_V8 = """DISAMBIGUATION: The model flagged >>> <<< as potential PII but is uncertain.

{context}

PREVIOUS: {prev_label}

CONFIRM as PII if the token is:
- Part of a name (given names, surnames, titles like Dr./Mr.)
- Part of an ID (credit card, SSN, passport, phone, email)
- Part of a location (street, city, address numbers)
- Part of a date, time, or age
- Continuation of previous entity (use I-X tag)

REJECT as O only if:
- Clearly a common word (the, and, is, with, for)
- Punctuation not part of an entity
- Generic text with no personal identification value

{valid_labels_str}

{{"label": "TAG"}}"""


PROMPT_V7 = """Classify the marked token (>>> <<<) for PII detection.

{context}

PREVIOUS TOKEN TAG: {prev_label}

PII CATEGORIES:
- PERSON: GIVENNAME, SURNAME, TITLE (names like John, Smith, Dr.)
- IDS: CREDITCARDNUMBER, SOCIALNUM, PASSPORTNUM, IDCARDNUM, DRIVERLICENSENUM, TAXNUM
- CONTACT: EMAIL, TELEPHONENUM
- LOCATION: STREET, CITY, BUILDINGNUM, ZIPCODE
- TIME: DATE, TIME, AGE
- DEMOGRAPHIC: SEX, GENDER

BIO TAGGING:
- B-X = Begin entity type X
- I-X = Inside/continuation of entity X
- O = Not PII (common words, punctuation, prepositions)

DECISION:
1. If previous is B-X or I-X and token continues same entity → I-X
2. If token is clearly identifiable personal information → B-X
3. If token is common word, punctuation, or not personal data → O

{valid_labels_str}

{{"label": "YOUR_TAG"}}"""


PROMPT_V6 = """Classify the target token as PII or not.

CONTEXT: {context}
PREVIOUS TAG: {prev_label}

BIO RULES:
- If previous is B-X or I-X and target continues the entity → I-X
- If previous is O and target is PII → B-X
- If not PII → O

VALID LABELS: {valid_labels_str}

OUTPUT JSON:
{{"label": "TAG"}}"""


PROMPT_V5 = """You are a PII classification expert. Verify or correct the model's prediction for a single token.

CONTEXT: {context}
TARGET: >>> {target_token} <<<
PREDICTION: {current_pred}
PREVIOUS: {prev_label}

RULES (priority order):

1. BIO CONSISTENCY (Highest)
   - Previous B-X/I-X + continues entity → I-X
   - Previous O → Only B-X or O (never I-X)
   - "John"[B-GIVENNAME] + "athan" → I-GIVENNAME
   - "555"[B-TELEPHONENUM] + "-1234" → I-TELEPHONENUM

2. NUMERIC PII (High Confidence)
   - CREDITCARD: 13-19 digits (possibly spaced/dashed)
   - PHONE: 7-15 digits, +country code, spaces/dashes/parentheses
   - SSN/SOCIALNUM: 9-11 digits with dashes (XXX-XX-XXXX)
   - EMAIL: user@domain.tld pattern → B-EMAIL/I-EMAIL
   - IBAN/ACCOUNT: alphanumeric 15-34 chars
   - IP: dotted decimal (192.168.x.x) or IPv6 → B-IP/I-IP

3. NAMES (Conservative)
   - CONFIRM: Capitalized in greeting/signature ("Dear John", "Sincerely, Mary")
   - REJECT: Sentence-start capitalization, job titles, organizations
   - Multi-token names: "Mary" [B-GIVENNAME] + "Jane" → I-GIVENNAME (same entity continues)
   - "Mario" [B-GIVENNAME] + "Rossi" → B-SURNAME (different entity type)

4. LOCATIONS (Strict)
   - STREET: Only with markers (Via, Street, Rd, Ave, Boulevard, Str.)
   - CITY: Proper city names only, not "the city"
   - Full address: "Via Roma 15" → B-STREET I-STREET I-STREET

5. DEFAULT: Trust model unless clear evidence against

VALID LABELS: {valid_labels_str}

OUTPUT (JSON only):
{{"reasoning": "Rule X: brief explanation (max 20 words)", "corrected_label": "TAG"}}

EXAMPLES:
{{"reasoning": "Rule 1: Dash continues phone entity", "corrected_label": "I-TELEPHONENUM"}}
{{"reasoning": "Rule 2: 16-digit credit card pattern", "corrected_label": "B-CREDITCARDNUMBER"}}
{{"reasoning": "Rule 3: Job title not a name", "corrected_label": "O"}}
"""

PROMPT_V3 = """TASK: Classify the target token with correct BIO-tag for PII detection.

CONTEXT: {context}
TARGET: >>> {target_token} <<<
PREVIOUS TAG: {prev_label}
CURRENT PREDICTION: {current_pred}

RULES:
1. CONTINUATION (I-tag): Use I-X if target continues the same entity type from previous token
- "John" [B-GIVENNAME] + "athan" → I-GIVENNAME
- "Via" [B-STREET] + "Roma" → I-STREET

2. NEW ENTITY (B-tag): Use B-X only for first token of a NEW entity
- "Mario" [B-GIVENNAME] + "Rossi" → B-SURNAME (different entity)

3. NON-PII (O): Use O for stopwords, articles, common verbs, punctuation
- "the", "and", "is", ",", "." → O

VALID LABELS:
{valid_labels_str}

OUTPUT JSON:
{{
"reasoning": "Brief explanation (max 20 words)",
"corrected_label": "TAG"
}}"""

PROMPT_V4 = """TASK: PII Classification & Correction.
Analyze the Target Token in context. The Baseline Model may be wrong.

CONTEXT: {context}
TARGET: >>> {target_token} <<<
PREVIOUS TAG: {prev_label}
CURRENT PREDICTION: {current_pred}

CRITICAL RULES (Based on Error Analysis):
1. RECALL MODE (If Prediction is O):
   - Actively check for missed numeric PII: Credit Cards (13-19 digits), Phone Numbers, IDs.
   - Check for missed Names (capitalized words in header/signature blocks).
   - If pattern matches PII strongly, OVERRIDE 'O'.

2. PRECISION MODE (Streets & Locations):
   - B-STREET/I-STREET: Only predict if explicit address markers exist (e.g., "Via", "St.", "Road", "Square").
   - Do not incorrectly classify City names or generic locations as Streets.

3. BIO CONSISTENCY:
   - "John" [B-GIVENNAME] + "Smith" → B-SURNAME (New entity part)
   - "New" [B-CITY] + "York" → I-CITY (Continuation)

VALID LABELS:
{valid_labels_str}

OUTPUT JSON:
{{
"reasoning": "Concise proof (e.g., 'Matches Luhn algo', 'No street marker')",
"corrected_label": "TAG"
}}"""


# Default prompt (V11: chain-of-thought reasoning for optimal performance)
PROMPT = PROMPT_V11
