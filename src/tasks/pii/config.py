"""
PII task configuration for NerGuard.

Bundles all PII-specific routing parameters into a RouteConfig instance.
These values were optimized via grid search (900 configurations, bootstrap CI)
on the AI4Privacy validation set.
"""

from src.core.route_config import RouteConfig

# Valid PII labels (BIO scheme) — 20 entity classes
VALID_PII_LABELS = [
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

# Entity classes without BIO prefix
PII_ENTITY_CLASSES = {
    lbl.replace("B-", "").replace("I-", "")
    for lbl in VALID_PII_LABELS if lbl != "O"
}
PII_ENTITY_CLASSES_WITH_O = PII_ENTITY_CLASSES | {"O"}

# Entity types that should be validated against regex (precision filter)
PII_REGEX_VALIDATABLE = {"SOCIALNUM"}

# Per-entity threshold overrides (more aggressive for numeric patterns)
PII_ENTITY_THRESHOLDS = {
    "CREDITCARDNUMBER": {"entropy": 0.4, "confidence": 0.9},
    "TELEPHONENUM": {"entropy": 0.5, "confidence": 0.85},
    "SOCIALNUM": {"entropy": 0.4, "confidence": 0.9},
    "DEFAULT": {"entropy": 0.583, "confidence": 0.787},
}


def get_pii_route_config() -> RouteConfig:
    """Create a RouteConfig for PII detection.

    Current configuration: all 20 entity types routable for both B- and I- tokens,
    no types blocked. This was the empirically best configuration (delta F1 +0.0222).
    """
    return RouteConfig(
        entropy_threshold=0.583,
        confidence_threshold=0.787,
        routable_entities=PII_ENTITY_CLASSES,
        blocked_entities=set(),
        routable_i_entities=PII_ENTITY_CLASSES,
        entity_thresholds=PII_ENTITY_THRESHOLDS,
        enable_selective=True,
        block_continuation_tokens=True,
        o_entropy_multiplier=1.5,
    )
