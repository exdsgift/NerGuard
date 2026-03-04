"""
Constants and configuration values for NerGuard.

This module centralizes all configuration values, thresholds, and label mappings
used throughout the project.
"""

DEFAULT_MODEL_PATH = "./models/mdeberta-pii-safe/final"
DEFAULT_DATA_PATH = "./data/processed/tokenized_data"
DEFAULT_LABEL_PATH = "./data/processed/id2label.json"

# THRESHOLDS (optimized via grid search in optimization/threshold_optimizer.py)
DEFAULT_ENTROPY_THRESHOLD = 0.583
DEFAULT_CONFIDENCE_THRESHOLD = 0.787

# SLIDING WINDOW PARAMETERS
MAX_CONTEXT_LENGTH = 512
OVERLAP = 128
STRIDE = MAX_CONTEXT_LENGTH - 2 - OVERLAP  # 382

# LLM CONFIGURATION
DEFAULT_LLM_SOURCE = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_CACHE_SIZE = 1000


# REGEX DEMOTION — entity types validated against regex patterns (precision filter).
# If the model predicts one of these but the text doesn't match the regex, demote to O.
REGEX_VALIDATABLE_ENTITIES = {"SOCIALNUM"}

# PER-ENTITY CONFIDENCE GATES — controls when LLM's "not PII" verdict is rejected.
# None = always accept LLM correction; value = reject LLM's O verdict if model conf > value.
# CAUTION: Gating preserves model predictions, so it increases recall but can hurt precision.
ENTITY_CONFIDENCE_GATES = {
    "DEFAULT": None,            # no gating — trust LLM corrections by default
}

# Entity-specific thresholds (more aggressive for numeric, conservative for names)
ENTITY_THRESHOLDS = {
    # Entities where LLM helps -> more aggressive routing
    "CREDITCARDNUMBER": {"entropy": 0.4, "confidence": 0.9},
    "TELEPHONENUM": {"entropy": 0.5, "confidence": 0.85},
    "SOCIALNUM": {"entropy": 0.4, "confidence": 0.9},
    # Default thresholds
    "DEFAULT": {"entropy": DEFAULT_ENTROPY_THRESHOLD, "confidence": DEFAULT_CONFIDENCE_THRESHOLD},
}

# VALID PII LABELS (BIO scheme)
VALID_LABELS = [
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
    "B-GENDER", "I-GENDER",
]

VALID_LABELS_SET = set(VALID_LABELS)

# Entity classes without BIO prefix — used by PROMPT_V13 paradigm
# where LLM predicts only the class and B-/I- is assigned deterministically
ENTITY_CLASSES = {lbl.replace("B-", "").replace("I-", "") for lbl in VALID_LABELS if lbl != "O"}
ENTITY_CLASSES_WITH_O = ENTITY_CLASSES | {"O"}

# SELECTIVE ENTITY ROUTING — Universal configuration (empirically best: ΔF1 +0.0222)
# All entity types routable for B- tokens; all I- tokens routable (no per-type filtering).
# See docs/notes/routing_observations.md for full ablation analysis.
ROUTABLE_ENTITIES = ENTITY_CLASSES        # all 20 types
BLOCKED_ENTITIES: set = set()             # no types blocked
ROUTABLE_I_ENTITIES = ENTITY_CLASSES      # I- routing open for all

# NVIDIA DATASET LABEL MAPPING
# Full map (original behavior) — includes ambiguous/merged entries.
# Use CLEAN_NVIDIA_MAP for unbiased evaluation.

NVIDIA_TO_MODEL_MAP = {
    # Direct Matches - Person
    "first_name": "GIVENNAME",
    "last_name": "SURNAME",
    "middle_name": "GIVENNAME",
    "name": "GIVENNAME",
    "user_name": "GIVENNAME",
    # Contact
    "email": "EMAIL",
    "phone_number": "TELEPHONENUM",
    "cell_phone": "TELEPHONENUM",
    "fax_number": "TELEPHONENUM",
    # Location
    "city": "CITY",
    "street_address": "STREET",
    "zipcode": "ZIPCODE",
    "postcode": "ZIPCODE",
    # IDs
    "ssn": "SOCIALNUM",
    "social_security_number": "SOCIALNUM",
    "tax_id": "TAXNUM",
    "driver_license": "DRIVERLICENSENUM",
    "drivers_license": "DRIVERLICENSENUM",
    "national_id": "IDCARDNUM",
    "passport_number": "PASSPORTNUM",
    # Financial
    "credit_debit_card": "CREDITCARDNUMBER",
    # Temporal
    "date": "DATE",
    "date_of_birth": "DATE",
    "date_time": "DATE",
    "time": "TIME",
    # Demographics
    "age": "AGE",
    "gender": "GENDER",
    "sexuality": "GENDER",
    # Excluded (Mapped to O - not PII in our schema)
    "company_name": "O",
    "organization": "O",
    "occupation": "O",
    "url": "O",
    "ip_address": "O",
    "ipv4": "O",
    "ipv6": "O",
    "country": "O",
    "state": "O",
    "account_number": "O",
}

# CLEAN NVIDIA LABEL MAPPING — unambiguous 1:1 mappings only.
# Used with unbiased=True in HybridEvaluator to avoid merging bias.
# Labels NOT listed here (nor in this dict's values) are in EXCLUDED_NVIDIA_LABELS
# and will be marked as -100 (excluded from TP/FP/FN calculations).
CLEAN_NVIDIA_MAP = {
    # Person
    "first_name": "GIVENNAME",
    "last_name": "SURNAME",
    # Contact
    "email": "EMAIL",
    "phone_number": "TELEPHONENUM",
    "cell_phone": "TELEPHONENUM",           # cell phone = telephone number
    # Location
    "city": "CITY",
    "street_address": "STREET",
    "zipcode": "ZIPCODE",
    "postcode": "ZIPCODE",                  # alias
    # IDs
    "ssn": "SOCIALNUM",
    "social_security_number": "SOCIALNUM",  # alias
    "tax_id": "TAXNUM",
    "driver_license": "DRIVERLICENSENUM",
    "drivers_license": "DRIVERLICENSENUM",  # alias
    "national_id": "IDCARDNUM",
    "passport_number": "PASSPORTNUM",
    # Financial
    "credit_debit_card": "CREDITCARDNUMBER",
    # Temporal
    "date": "DATE",
    "date_of_birth": "DATE",
    "time": "TIME",
    # Demographics
    "age": "AGE",
    "gender": "GENDER",
    # Non-PII in model schema — valid O ground truth, included to avoid FP inflation
    "company_name": "O",
    "organization": "O",
    "occupation": "O",
    "url": "O",
    "ip_address": "O",
    "ipv4": "O",
    "ipv6": "O",
    "country": "O",
    "state": "O",
    "account_number": "O",
}

# NVIDIA labels excluded from unbiased evaluation.
# Tokens with these labels are marked as -100 (no contribution to TP/FP/FN).
# Rationale: ambiguous or semantically incorrect mapping to model classes.
EXCLUDED_NVIDIA_LABELS = {
    "middle_name",  # Middle name ≠ given name; model not trained on this distinction
    "name",         # Too generic (could be given name, surname, company name)
    "user_name",    # Username ≠ given name; semantically incorrect original merge
    "fax_number",   # Rare in training corpus, ambiguous context vs phone
    "date_time",    # Compound datetime span → token alignment issues
    "sexuality",    # Sexuality ≠ gender; semantically incorrect original merge
}

# NVIDIA ALIAS → BASE MODEL LABEL MAPPING
# Used in LLMRouter._validate_response() to accept NVIDIA-style entity names.
# Only includes NVIDIA labels that map to non-O base model entities (FP-safe):
# labels that map to O in CLEAN_NVIDIA_MAP (account_number, ip_address, url,
# organization, country, state) are intentionally excluded so the LLM never
# sees them in the prompt and cannot produce them (no FP inflation).
NVIDIA_CLASS_TO_BASE = {
    # Person
    "first_name": "GIVENNAME",
    "last_name": "SURNAME",
    "middle_name": "GIVENNAME",
    # Contact
    "phone_number": "TELEPHONENUM",
    "cell_phone": "TELEPHONENUM",
    "fax_number": "TELEPHONENUM",
    # Location
    "street_address": "STREET",
    "zipcode": "ZIPCODE",
    "postcode": "ZIPCODE",
    # IDs
    "ssn": "SOCIALNUM",
    "social_security_number": "SOCIALNUM",
    "tax_id": "TAXNUM",
    "driver_license": "DRIVERLICENSENUM",
    "drivers_license": "DRIVERLICENSENUM",
    "certificate_license_number": "DRIVERLICENSENUM",
    "national_id": "IDCARDNUM",
    "passport_number": "PASSPORTNUM",
    # Financial
    "credit_debit_card": "CREDITCARDNUMBER",
    # Temporal & Demographics
    "date_of_birth": "DATE",
    "sexuality": "GENDER",
}

# Extended FP-safe entity class set: base 20 classes + NVIDIA PII aliases.
# Used to expand the LLM's valid response vocabulary in V16_SPAN routing.
EXTENDED_ENTITY_CLASSES_WITH_O = ENTITY_CLASSES_WITH_O | set(NVIDIA_CLASS_TO_BASE.keys())

# UNIFIED SCHEMA FOR CROSS-MODEL BENCHMARKS
UNIFIED_SCHEMA = {
    "PERSON": ["GIVENNAME", "SURNAME", "TITLE"],
    "LOCATION": ["CITY", "STREET", "BUILDINGNUM", "ZIPCODE"],
    "PHONE_NUMBER": ["TELEPHONENUM"],
    "EMAIL_ADDRESS": ["EMAIL"],
    "DATE_TIME": ["DATE", "TIME"],
    "GOV_ID": ["SOCIALNUM", "PASSPORTNUM", "DRIVERLICENSENUM", "IDCARDNUM", "TAXNUM"],
    "CREDIT_CARD": ["CREDITCARDNUMBER"],
    "AGE": ["AGE"],
    "GENDER": ["GENDER", "SEX"],
}

# Reverse mapping: specific label -> unified category
LABEL_TO_UNIFIED = {}
for unified_cat, specific_labels in UNIFIED_SCHEMA.items():
    for label in specific_labels:
        LABEL_TO_UNIFIED[label] = unified_cat
        LABEL_TO_UNIFIED[f"B-{label}"] = unified_cat
        LABEL_TO_UNIFIED[f"I-{label}"] = unified_cat

# CONLL-2003 LABEL MAPPING (for standard NER benchmark evaluation)
CONLL_ID_TO_LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}

CONLL_TO_UNIFIED = {
    "PER": "PER",
    "LOC": "LOC",
    "ORG": "O",     # Organizations not in PII schema
    "MISC": "O",    # Miscellaneous not in PII schema
}

# TRAINING CONFIGURATION
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BASE_MODEL = "microsoft/mdeberta-v3-base"
