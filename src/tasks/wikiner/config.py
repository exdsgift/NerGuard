"""
General NER task configuration for NerGuard — WikiNeural compatibility experiment.

This configuration intentionally disables all PII-specific validators and uses
a compact general-purpose label schema (PER, ORG, LOC, MISC) to test the
selective routing principle in isolation.

This is the "minimal general NER compatibility mode" described in the thesis
(Section: Cross-Domain Generalization, Supervisor suggestion iv).

Model: dslim/bert-base-NER (CoNLL-2003, trained on PER/ORG/LOC/MISC)
Dataset: Babelscape/wikineural (same label schema, multilingual)
Validator: PassthroughValidator (no PII-specific structural rules)
"""

from src.core.route_config import RouteConfig

# WikiNeural / CoNLL-style entity classes
WIKINER_ENTITY_CLASSES = {"PER", "ORG", "LOC", "MISC"}
WIKINER_ENTITY_CLASSES_WITH_O = WIKINER_ENTITY_CLASSES | {"O"}

# No per-entity regex validation for general NER
WIKINER_REGEX_VALIDATABLE: set = set()

# Default thresholds — same starting point as PII/biomedical baselines
WIKINER_ENTITY_THRESHOLDS = {
    "DEFAULT": {"entropy": 0.583, "confidence": 0.787},
}

# HuggingFace model
WIKINER_MODEL = "dslim/bert-base-NER"


def get_wikiner_route_config() -> RouteConfig:
    """Create a RouteConfig for general NER (WikiNeural).

    No entity-specific overrides, no blocked entities, PassthroughValidator.
    This is the cleanest possible routing configuration to test the routing
    principle without any domain-specific engineering.
    """
    return RouteConfig(
        entropy_threshold=0.583,
        confidence_threshold=0.787,
        routable_entities=WIKINER_ENTITY_CLASSES,
        blocked_entities=set(),
        routable_i_entities=WIKINER_ENTITY_CLASSES,
        entity_thresholds=WIKINER_ENTITY_THRESHOLDS,
        enable_selective=True,
        block_continuation_tokens=True,
        o_entropy_multiplier=1.5,
    )
