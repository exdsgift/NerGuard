import json

VALID_LABELS_DICT = [
    "O",
    "B-GIVENNAME",
    "I-GIVENNAME",
    "B-SURNAME",
    "I-SURNAME",
    "B-TITLE",
    "I-TITLE",
    "B-CITY",
    "I-CITY",
    "B-STREET",
    "I-STREET",
    "B-BUILDINGNUM",
    "I-BUILDINGNUM",
    "B-ZIPCODE",
    "I-ZIPCODE",
    "B-IDCARDNUM",
    "I-IDCARDNUM",
    "B-PASSPORTNUM",
    "I-PASSPORTNUM",
    "B-DRIVERLICENSENUM",
    "I-DRIVERLICENSENUM",
    "B-SOCIALNUM",
    "I-SOCIALNUM",
    "B-TAXNUM",
    "I-TAXNUM",
    "B-CREDITCARDNUMBER",
    "I-CREDITCARDNUMBER",
    "B-EMAIL",
    "I-EMAIL",
    "B-TELEPHONENUM",
    "I-TELEPHONENUM",
    "B-DATE",
    "I-DATE",
    "B-TIME",
    "I-TIME",
    "B-AGE",
    "I-AGE",
    "B-SEX",
    "I-SEX",
    "B-GENDER",
    "I-GENDER",
]

VALID_LABELS_STR = json.dumps(VALID_LABELS_DICT, indent=2)

PROMPT_V1 = """
### ROLE
You are a Data Privacy Auditor specializing in fixing broken BIO entity tags.

### TASK
Analyze the **Target Token** (inside `>>> <<<`) and assign the correct PII Label.
You MUST consider the **Previous Label** to decide between B- (Beginning) and I- (Inside/Continuation).

### INPUT DATA
Context: "{context}"
Target Token: "{target_token}"
Previous token BIO Label: "{prev_label}"

### STRICT BIO RULES
1. **CONTINUITY CHECK**: 
   - If `Previous Label` was `B-TAG` and the target is part of that ID (like a number or hyphen), result MUST be `I-TAG`.
   - **NEVER** output `B-TAG` immediately after `B-TAG` or `I-TAG` of the same type (e.g. `B-IDNUM` -> `B-IDNUM` is WRONG. Use `I-IDNUM`).

2. **CATEGORY BAN**: 
   - Do NOT use category names like "IDS", "OTHER", "PEOPLE" as labels.
   - Use ONLY the specific tags from the list below (e.g., "B-IDNUM", "O").

3. **FRAGMENT HANDLING**:
   - Symbols like "-" or "/" inside dates/IDs are usually `I-TAG`.

### VALID LABELS LIST
{valid_labels_str}

### RESPONSE FORMAT (JSON)
{{
   "corrected_label": "TAG"
}}
"""

PROMPT_V2 = """
### ROLE
You are a Linguistic Expert specialized in Named Entity Recognition (NER) and PII protection.
Your task is to correct misclassified BIO tags based on strict grammatical and contextual rules.

### INPUT DATA
- **Context**: "{context}"
- **Target Token**: >>> {target_token} <<< (The token to classify)
- **Previous Label**: "{prev_label}" (The tag assigned to the word immediately before the target)

### CRITICAL RULES (BIO SCHEME)
1. **CONSISTENCY (I-TAG)**:
   - If the *Target Token* is a continuation of the entity in *Previous Label*, you MUST use the `I-` version of that tag.
   - *Example*: If Prev is `B-GIVENNAME` ("John") and Target is ("athan"), Target MUST be `I-GIVENNAME`.
   - *Example*: If Prev is `B-STREET` ("Via") and Target is ("Roma"), Target MUST be `I-STREET`.

2. **NEW ENTITY (B-TAG)**:
   - Use `B-` ONLY if the token marks the *start* of a completely new entity.
   - *Example*: "Mario [B-GIVENNAME] Rossi [B-SURNAME]" -> Distinct entities.

3. **NON-PII (O)**:
   - If the token is a stopword, punctuation (that isn't part of an ID), or a common verb/noun, use "O".

### VALID LABELS
{valid_labels_str}

### INSTRUCTIONS
1. Analyze the **Context** to understand the semantic meaning of the token.
2. Check the **Previous Label** to enforce BIO continuity.
3. Provide a short "reasoning" explaining why it is Start (B), Continuation (I), or Outside (O).
4. Output the final JSON.

### RESPONSE FORMAT
{{
   "reasoning": "Brief explanation of grammar/context...",
   "corrected_label": "TAG"
}}
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

PROMPT_V5 = """You are a PII classification expert. Your ONLY job is to verify or correct the model's prediction for a single token.

=== INPUT DATA ===
CONTEXT: {context}
TARGET TOKEN: >>> {target_token} <<<
MODEL PREDICTION: {current_pred}
PREVIOUS TOKEN LABEL: {prev_label}

=== YOUR TASK ===
1. Examine the TARGET TOKEN in the CONTEXT
2. Decide if MODEL PREDICTION is correct OR needs correction
3. IMPORTANT: Trust the model UNLESS you have strong evidence it's wrong

=== DECISION RULES (in priority order) ===

**RULE 1: BIO CONSISTENCY (Highest Priority)**
- If PREVIOUS LABEL was "B-X" or "I-X" AND target token continues that entity → MUST use "I-X"
- If PREVIOUS LABEL was "O" → Can ONLY use "B-X" or "O" (NEVER "I-X")
- Examples:
  * "John" [B-GIVENNAME] + "athan" → MUST be I-GIVENNAME
  * "555" [B-TELEPHONENUM] + "-" → MUST be I-TELEPHONENUM
  * "New" [B-CITY] + "York" → MUST be I-CITY

**RULE 2: NUMERIC PII PATTERNS (High Confidence)**
When MODEL PREDICTION suggests numeric PII, verify pattern:
- CREDIT CARDS: 13-19 consecutive digits, possibly with spaces/dashes
- PHONE NUMBERS: 7-15 digits with optional country code, spaces, dashes, parentheses
- SSN/SOCIAL: 9-11 digits, often with dashes
- TAX ID: 8-15 digits
- ZIPCODE: 5 or 9 digits
→ If pattern matches strongly, CONFIRM the prediction even if context is ambiguous

**RULE 3: NAMES (Be Conservative)**
For GIVENNAME/SURNAME predictions:
- CONFIRM if: Capitalized word in greeting/signature/form fields ("Dear John", "Sincerely, Mary")
- REJECT if: Common word that happens to be capitalized at sentence start
- REJECT if: Job title, organization name, or generic noun
- Remember that multiple languages/cultures may have different name conventions

**RULE 4: LOCATIONS (Context Required)**
- B-STREET/I-STREET: ONLY if explicit address markers exist ("Via", "Street", "Road", "Avenue", "Boulevard", number + street name)
- B-CITY/I-CITY: Proper city names, but NOT generic "the city"
- B-ZIPCODE: Only valid postal codes

**RULE 5: DEFAULT (Trust Model)**
- If none of the above rules give clear evidence → Keep MODEL PREDICTION unchanged
- Don't overcorrect based on weak signals

=== VALID LABELS ===
{valid_labels_str}

=== OUTPUT FORMAT (JSON ONLY) ===
{{
  "reasoning": "Brief explanation: which rule applied and why (max 25 words)",
  "corrected_label": "TAG"
}}

=== EXAMPLES ===
Example 1 (BIO Consistency):
Context: "Call me at 555-1234"
Target: "-" | Prediction: B-TELEPHONENUM | Previous: B-TELEPHONENUM
→ {{"reasoning": "Rule 1: Dash continues phone number entity", "corrected_label": "I-TELEPHONENUM"}}

Example 2 (Trust Model for Names):
Context: "Dear Smith, thank you"
Target: "Smith" | Prediction: B-SURNAME | Previous: O
→ {{"reasoning": "Rule 3: Capitalized name in greeting, confirms model", "corrected_label": "B-SURNAME"}}

Example 3 (Numeric Pattern):
Context: "Card number 4532015112830366"
Target: "4532015112830366" | Prediction: B-CREDITCARDNUMBER | Previous: O
→ {{"reasoning": "Rule 2: 16-digit pattern matches credit card", "corrected_label": "B-CREDITCARDNUMBER"}}

Example 4 (Conservative on Ambiguous Names):
Context: "The company Director spoke"
Target: "Director" | Prediction: O | Previous: O
→ {{"reasoning": "Rule 3: Job title, not a name, confirm O", "corrected_label": "O"}}
"""

PROMPT = PROMPT_V5
