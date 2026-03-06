"""
Task-agnostic routing configuration for NerGuard.

RouteConfig encapsulates all routing decisions: which entities to route,
which to block, and what uncertainty thresholds to use. This decouples
the routing logic from any specific NER task (PII, biomedical, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class RouteConfig:
    """Configuration for the uncertainty-based entity router.

    Attributes:
        entropy_threshold: Default Shannon entropy threshold above which
            a prediction is considered uncertain.
        confidence_threshold: Default softmax confidence threshold below
            which a prediction is considered uncertain.
        routable_entities: Entity types that may be routed for B- tokens.
            If None, all entity types are routable.
        blocked_entities: Entity types that are never routed (empirically harmful).
        routable_i_entities: Entity types where I- (continuation) token routing
            is also allowed. By default, I- tokens are blocked because LLMs
            struggle with BIO sequence constraints.
        entity_thresholds: Per-entity threshold overrides.
            Maps entity type -> {"entropy": float, "confidence": float}.
        enable_selective: Whether to apply entity-type filtering.
            When False, all uncertain predictions are routed regardless of type.
        block_continuation_tokens: Whether to block I- token routing by default.
        o_entropy_multiplier: Multiplier applied to entropy threshold for
            O-predicted tokens (higher = more conservative false-negative recovery).
    """

    entropy_threshold: float = 0.583
    confidence_threshold: float = 0.787
    routable_entities: Optional[Set[str]] = None
    blocked_entities: Set[str] = field(default_factory=set)
    routable_i_entities: Optional[Set[str]] = None
    entity_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    enable_selective: bool = True
    block_continuation_tokens: bool = True
    o_entropy_multiplier: float = 1.5
