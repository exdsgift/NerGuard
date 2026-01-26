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
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_CACHE_SIZE = 1000

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
    # Financial
    "credit_debit_card": "CREDITCARDNUMBER",
    # Temporal
    "date": "DATE",
    "date_of_birth": "DATE",
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
    "national_id": "O",
    "passport_number": "O",
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

# TRAINING CONFIGURATION
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BASE_MODEL = "microsoft/mdeberta-v3-base"
