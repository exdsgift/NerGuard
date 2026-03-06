"""
Biomedical NER task configuration for NerGuard.

BC5CDR dataset: Chemical + Disease entity recognition on PubMed abstracts.
Model: tner/roberta-large-bc5cdr (F1=0.884).
"""

from src.core.route_config import RouteConfig

# BC5CDR label scheme
VALID_BIO_LABELS = [
    "O",
    "B-Chemical", "I-Chemical",
    "B-Disease", "I-Disease",
]

BIO_ENTITY_CLASSES = {"Chemical", "Disease"}
BIO_ENTITY_CLASSES_WITH_O = BIO_ENTITY_CLASSES | {"O"}

# No per-entity regex validation for biomedical NER
BIO_REGEX_VALIDATABLE = set()

# Default thresholds — start with same as PII, can be optimized later
BIO_ENTITY_THRESHOLDS = {
    "DEFAULT": {"entropy": 0.583, "confidence": 0.787},
}

# HuggingFace identifiers
BC5CDR_DATASET = "tner/bc5cdr"
BC5CDR_MODEL = "tner/roberta-large-bc5cdr"

# Label mapping from dataset integer tags to BIO strings
BC5CDR_TAG_TO_LABEL = {
    0: "O",
    1: "B-Chemical",
    2: "B-Disease",
    3: "I-Disease",
    4: "I-Chemical",
}


def get_biomedical_route_config() -> RouteConfig:
    """Create a RouteConfig for biomedical NER (BC5CDR).

    Both Chemical and Disease are routable. No entities blocked initially.
    """
    return RouteConfig(
        entropy_threshold=0.583,
        confidence_threshold=0.787,
        routable_entities=BIO_ENTITY_CLASSES,
        blocked_entities=set(),
        routable_i_entities=BIO_ENTITY_CLASSES,
        entity_thresholds=BIO_ENTITY_THRESHOLDS,
        enable_selective=True,
        block_continuation_tokens=True,
        o_entropy_multiplier=1.5,
    )
