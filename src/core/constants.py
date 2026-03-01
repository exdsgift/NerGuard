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

# SELECTIVE ENTITY ROUTING
# Entities where LLM routing has proven beneficial (numeric patterns)
ROUTABLE_ENTITIES = {
    "CREDITCARDNUMBER",
    "TELEPHONENUM",
    "SOCIALNUM",
    "DATE",
    "TAXNUM",
    "PASSPORTNUM",
    "DRIVERLICENSENUM",
    "IDCARDNUM",
}

# Entities where LLM routing causes harm (name confusion, BIO errors, high baseline)
BLOCKED_ENTITIES = {
    "GIVENNAME",
    "SURNAME",
    "TITLE",
    "CITY",
    "STREET",
    "EMAIL",
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
# NVIDIA DATASET LABEL MAPPING

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
