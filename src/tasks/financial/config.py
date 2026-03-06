"""
Financial NER task configuration for NerGuard.

BUSTER dataset: Business Entity Recognition on SEC filings (M&A documents).
6 entity classes: buying/selling/acquired companies, legal/generic advisors, revenues.
Model: whataboutyou-ai/financial_bert (BERT-base fine-tuned, F1=0.825).
"""

from src.core.route_config import RouteConfig

# BUSTER model and dataset
BUSTER_MODEL = "whataboutyou-ai/financial_bert"
BUSTER_DATASET = "expertai/BUSTER"

# Entity classes (6 types from M&A domain)
BUSTER_ENTITY_CLASSES = {
    "Advisors.GENERIC_CONSULTING_COMPANY",
    "Advisors.LEGAL_CONSULTING_COMPANY",
    "Generic_Info.ANNUAL_REVENUES",
    "Parties.ACQUIRED_COMPANY",
    "Parties.BUYING_COMPANY",
    "Parties.SELLING_COMPANY",
}

BUSTER_ENTITY_CLASSES_WITH_O = BUSTER_ENTITY_CLASSES | {"O"}

# Short aliases for display/prompt readability
BUSTER_SHORT_NAMES = {
    "Advisors.GENERIC_CONSULTING_COMPANY": "Generic Advisor",
    "Advisors.LEGAL_CONSULTING_COMPANY": "Legal Advisor",
    "Generic_Info.ANNUAL_REVENUES": "Annual Revenues",
    "Parties.ACQUIRED_COMPANY": "Acquired Company",
    "Parties.BUYING_COMPANY": "Buying Company",
    "Parties.SELLING_COMPANY": "Selling Company",
}

# Default thresholds — use --calibrate to auto-tune
BUSTER_ENTITY_THRESHOLDS = {
    "DEFAULT": {"entropy": 0.5, "confidence": 0.8},
}


def get_financial_route_config() -> RouteConfig:
    """Create a RouteConfig for financial NER (BUSTER).

    All 6 entity types are routable. Uses default thresholds initially;
    use --calibrate to auto-tune for this model.
    """
    return RouteConfig(
        entropy_threshold=0.5,
        confidence_threshold=0.8,
        routable_entities=BUSTER_ENTITY_CLASSES,
        blocked_entities=set(),
        routable_i_entities=BUSTER_ENTITY_CLASSES,
        entity_thresholds=BUSTER_ENTITY_THRESHOLDS,
        enable_selective=True,
        block_continuation_tokens=True,
        o_entropy_multiplier=1.5,
    )
