import json

VALID_LABELS_DICT = {
   "PEOPLE": ["B-GIVENNAME", "I-GIVENNAME", "B-SURNAME", "I-SURNAME", "B-TITLE"],
   "LOCATION": ["B-CITY", "I-CITY", "B-STREET", "I-STREET", "B-BUILDINGNUM", "I-BUILDINGNUM", "B-ZIPCODE", "I-ZIPCODE"],
   "IDS": ["B-IDNUM", "I-IDNUM", "B-PASSPORTNUM", "I-PASSPORTNUM", "B-DRIVERLICENSENUM", "I-DRIVERLICENSENUM", "B-SOCIALNUM", "I-SOCIALNUM", "B-TAXNUM", "I-TAXNUM"],
   "CONTACT": ["B-EMAIL", "I-EMAIL", "B-TELEPHONENUM", "I-TELEPHONENUM"],
   "OTHER_PII": ["B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-AGE", "I-AGE", "B-SEX", "I-SEX"]
}

VALID_LABELS_STR = json.dumps(VALID_LABELS_DICT, indent=2)

PROMPT = """
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