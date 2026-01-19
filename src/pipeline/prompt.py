import json

# VALID_LABELS_DICT = {
# "PEOPLE": [
#       "B-GIVENNAME", "I-GIVENNAME", 
#       "B-SURNAME", "I-SURNAME", 
#       "B-TITLE", "I-TITLE"
# ],
# "LOCATION": [
#       "B-CITY", "I-CITY", 
#       "B-STREET", "I-STREET", 
#       "B-BUILDINGNUM", "I-BUILDINGNUM", 
#       "B-ZIPCODE", "I-ZIPCODE"
# ],
# "IDS": [
#       "B-IDCARDNUM", "I-IDCARDNUM", 
#       "B-PASSPORTNUM", "I-PASSPORTNUM", 
#       "B-DRIVERLICENSENUM", "I-DRIVERLICENSENUM", 
#       "B-SOCIALNUM", "I-SOCIALNUM", 
#       "B-TAXNUM", "I-TAXNUM",
#       "B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER"
# ],
# "CONTACT": [
#       "B-EMAIL", "I-EMAIL", 
#       "B-TELEPHONENUM", "I-TELEPHONENUM"
# ],
# "OTHER_PII": [
#       "B-DATE", "I-DATE", 
#       "B-TIME", "I-TIME", 
#       "B-AGE", "I-AGE", 
#       "B-SEX", "I-SEX",
#       "B-GENDER", "I-GENDER"
# ]
# }

VALID_LABELS_DICT = [
         "O",
         "B-GIVENNAME", "I-GIVENNAME", 
         "B-SURNAME", "I-SURNAME", 
         "B-TITLE", "I-TITLE",
         "B-CITY", "I-CITY", 
         "B-STREET", "I-STREET", 
         "B-BUILDINGNUM", "I-BUILDINGNUM",
         "B-ZIPCODE", "I-ZIPCODE", 
         "B-IDCARDNUM", "I-IDCARDNUM", 
         "B-PASSPORTNUM", "I-PASSPORTNUM",
         "B-DRIVERLICENSENUM", "I-DRIVERLICENSENUM", 
         "B-SOCIALNUM", "I-SOCIALNUM",
         "B-TAXNUM", "I-TAXNUM",
         "B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER",
         "B-EMAIL", "I-EMAIL",
         "B-TELEPHONENUM", "I-TELEPHONENUM",
         "B-DATE", "I-DATE", 
         "B-TIME", "I-TIME", 
         "B-AGE", "I-AGE", 
         "B-SEX", "I-SEX", 
         "B-GENDER", "I-GENDER"
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

# * new prompt version to reduce over-correction

PROMPT_IMPROVED = """You are a PII (Personally Identifiable Information) detection expert.

**TASK**: Determine if the specific target token represents sensitive personal information.

**CONTEXT**: 
{context}

--- INPUT DATA ANALYSIS ---
1. **TARGET TOKEN**: {target_token}  <-- Analyze this
2. **MODEL PREDICTION**: {current_pred}  <-- Verify this
3. **PREVIOUS TOKEN LABEL**: {prev_label}  <-- Use ONLY for BIO consistency

**DECISION RULES**:

1. **IS IT ACTUALLY PII?**
   - PERSON: Names (First, Last).
   - LOCATION: Cities, addresses, zip codes, specific buildings.
   - CONTACT: Phone numbers, emails.
   - GOV_ID: SSN, Passport, Driver License, Tax ID.
   - OTHER: Ages, Genders, Job Titles (only if identifying).

2. **NOT PII (Common False Positives)**:
   - Common nouns/verbs (even if capitalized at start of sentence).
   - Generic place names (e.g., "the hospital", "town hall").
   - Company/Org names (unless they look exactly like a person's name).

3. **BIO LOGIC CHECK**:
   - IF Previous Label was 'B-TAG' or 'I-TAG' AND Current is the continuation of that entity → Force 'I-TAG'.
   - IF Previous Label was 'O' → Current MUST be 'B-TAG' (if PII) or 'O'.
   - NEVER use 'I-TAG' if Previous Label was 'O'.

**INSTRUCTIONS**:
- Trust the "MODEL PREDICTION" unless there is a clear semantic error (e.g. labeled City but is a Person).
- If the token is ambiguous (e.g. "Mark" as a name vs "mark" as a verb), use the CONTEXT.

**OUTPUT** (JSON only):
{{
   "reasoning": "Explain why it matches/mismatches PII definition and BIO logic",
   "corrected_label": "TAG"
}}

**VALID LABELS**: {valid_labels_str}
"""

PROMPT_MINIMAL = """Role: PII Expert (BIO Scheme).
Context: {context}

Target Token: >>> {target_token} <<<

--- DATA TIMELINE ---
Step 1 (History): Previous Label       = {prev_label}
Step 2 (Target):  Current Model Guess  = {current_pred}

Task: Verify 'Current Model Guess'.
Rules:
1. Use 'Previous Label' ONLY to check BIO consistency (e.g. B-PER -> I-PER).
2. Fix the guess only if context proves it wrong.
3. Valid Labels: {valid_labels_str}

Response (JSON):
{{
   "reasoning": "brief explanation",
   "corrected_label": "TAG"
}}
"""


PROMPT = PROMPT_V3