"""
Entity-Specific Router for NerGuard.

Routes predictions to LLM only for entity types where LLM corrections
have proven beneficial. The routing decision is based on:
1. Uncertainty thresholds (Shannon entropy, softmax confidence)
2. Entity type filtering (some types benefit from LLM, others don't)
3. BIO position (B- vs I- tokens may route differently)

This module is task-agnostic: routing configuration is supplied via
RouteConfig or constructor parameters.
"""

import logging
from typing import Dict, Optional, Set, Tuple

from src.core.route_config import RouteConfig

logger = logging.getLogger(__name__)


class EntitySpecificRouter:
    """
    Intelligent router that decides whether to invoke LLM based on:
    1. Uncertainty thresholds (entropy, confidence)
    2. Entity type (some types benefit from LLM, others don't)
    3. Per-entity-type threshold configuration

    Can be initialized from a RouteConfig or with individual parameters.

    Example:
        >>> from src.core.route_config import RouteConfig
        >>> config = RouteConfig(entropy_threshold=0.5, confidence_threshold=0.8)
        >>> router = EntitySpecificRouter.from_config(config)
        >>> router.should_route("B-DISEASE", entropy=0.8, confidence=0.6)
        True
    """

    def __init__(
        self,
        entropy_threshold: float = 0.583,
        confidence_threshold: float = 0.787,
        routable_entities: Optional[Set[str]] = None,
        blocked_entities: Optional[Set[str]] = None,
        routable_i_entities: Optional[Set[str]] = None,
        entity_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        enable_selective: bool = True,
        block_continuation_tokens: bool = True,
    ):
        """
        Initialize the EntitySpecificRouter.

        Args:
            entropy_threshold: Default entropy threshold for routing
            confidence_threshold: Default confidence threshold for routing
            routable_entities: Entity types routed for B- tokens. None = route all types.
            blocked_entities: Entity types never routed. None = block none.
            routable_i_entities: Entity types where I- routing is also allowed.
                None = same as routable_entities.
            entity_thresholds: Per-entity-type threshold overrides
            enable_selective: Whether to apply entity-type filtering (False = route all)
            block_continuation_tokens: Whether to apply I- continuation token filtering
        """
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.routable_entities = routable_entities
        self.blocked_entities = blocked_entities or set()
        self.routable_i_entities = routable_i_entities
        self.entity_thresholds = entity_thresholds or {
            "DEFAULT": {"entropy": entropy_threshold, "confidence": confidence_threshold}
        }
        self.enable_selective = enable_selective
        self.block_continuation_tokens = block_continuation_tokens

        # Statistics tracking
        self.stats = {
            "total_checked": 0,
            "routed": 0,
            "blocked_by_entity": 0,
            "blocked_by_continuation": 0,
            "blocked_by_threshold": 0,
            "routed_by_entity": {},
        }

    @classmethod
    def from_config(cls, config: RouteConfig) -> "EntitySpecificRouter":
        """Create a router from a RouteConfig dataclass."""
        return cls(
            entropy_threshold=config.entropy_threshold,
            confidence_threshold=config.confidence_threshold,
            routable_entities=config.routable_entities,
            blocked_entities=config.blocked_entities,
            routable_i_entities=config.routable_i_entities,
            entity_thresholds=config.entity_thresholds,
            enable_selective=config.enable_selective,
            block_continuation_tokens=config.block_continuation_tokens,
        )

    def should_route(
        self,
        predicted_label: str,
        entropy: float,
        confidence: float,
    ) -> bool:
        """
        Decide whether to route this prediction to LLM.

        The decision follows this logic:
        1. Check uncertainty: entropy > threshold AND confidence < threshold
        2. If not uncertain, don't route
        3. If selective routing disabled, route all uncertain predictions
        4. If I- continuation token and block_continuation_tokens=True, don't route
        5. If entity type is blocked (GIVENNAME, SURNAME), don't route
        6. If entity type is routable (CREDITCARD, TELEPHONE), route
        7. For 'O' predictions with very high uncertainty, route (catch false negatives)
        8. Default: don't route unknown entity types

        Args:
            predicted_label: Model's prediction (e.g., "B-SURNAME", "O")
            entropy: Shannon entropy of the prediction distribution
            confidence: Maximum probability (confidence) of the prediction

        Returns:
            True if the prediction should be routed to LLM, False otherwise
        """
        self.stats["total_checked"] += 1

        # Extract entity type from BIO label
        entity_type = self._extract_entity_type(predicted_label)

        # Get entity-specific thresholds or defaults
        thresholds = self.entity_thresholds.get(
            entity_type,
            self.entity_thresholds.get("DEFAULT", {
                "entropy": self.entropy_threshold,
                "confidence": self.confidence_threshold,
            })
        )

        ent_thresh = thresholds.get("entropy", self.entropy_threshold)
        conf_thresh = thresholds.get("confidence", self.confidence_threshold)

        # Check uncertainty thresholds
        is_uncertain = entropy > ent_thresh and confidence < conf_thresh

        if not is_uncertain:
            self.stats["blocked_by_threshold"] += 1
            return False

        # If selective routing is disabled, route all uncertain predictions
        if not self.enable_selective:
            self.stats["routed"] += 1
            return True

        # I- continuation token filtering.
        # Only CREDITCARDNUMBER and TELEPHONENUM benefit from I- routing with V13
        # (empirical: I-CREDITCARDNUMBER net +120, I-TELEPHONENUM net +76).
        # All other I- tokens are blocked: I-SURNAME(-16), I-GIVENNAME(-15),
        # I-DATE(-12), I-EMAIL(-6), I-STREET(-6).
        if self.block_continuation_tokens and predicted_label.startswith("I-"):
            if self.routable_i_entities is not None and entity_type not in self.routable_i_entities:
                self.stats["blocked_by_continuation"] += 1
                logger.debug(f"Routing blocked for I- continuation: {predicted_label}")
                return False
            # entity_type in routable_i_entities → fall through to entity routing check

        # Check if entity is blocked (causes harm)
        if entity_type in self.blocked_entities:
            self.stats["blocked_by_entity"] += 1
            logger.debug(f"Routing blocked for entity type: {entity_type}")
            return False

        # Check if entity is routable (provides benefit)
        # routable_entities=None means all types are routable
        if self.routable_entities is None or entity_type in self.routable_entities:
            self.stats["routed"] += 1
            self.stats["routed_by_entity"][entity_type] = (
                self.stats["routed_by_entity"].get(entity_type, 0) + 1
            )
            return True

        # For 'O' predictions, route only if very uncertain (1.5x threshold)
        # This helps catch false negatives where model missed an entity
        if predicted_label == "O" and entropy > ent_thresh * 1.5:
            self.stats["routed"] += 1
            self.stats["routed_by_entity"]["O_high_uncertainty"] = (
                self.stats["routed_by_entity"].get("O_high_uncertainty", 0) + 1
            )
            return True

        # Default: don't route unknown entity types
        self.stats["blocked_by_threshold"] += 1
        return False

    def _extract_entity_type(self, label: str) -> str:
        """
        Extract entity type from BIO label.

        Examples:
            'B-EMAIL' -> 'EMAIL'
            'I-SURNAME' -> 'SURNAME'
            'O' -> 'O'
        """
        if label == "O":
            return "O"
        if "-" in label:
            return label.split("-", 1)[1]
        return label

    def get_thresholds_for_entity(self, entity_type: str) -> Tuple[float, float]:
        """
        Get the entropy and confidence thresholds for a specific entity type.

        Args:
            entity_type: Entity type (without B-/I- prefix)

        Returns:
            Tuple of (entropy_threshold, confidence_threshold)
        """
        thresholds = self.entity_thresholds.get(
            entity_type,
            self.entity_thresholds.get("DEFAULT", {
                "entropy": self.entropy_threshold,
                "confidence": self.confidence_threshold,
            })
        )
        return thresholds.get("entropy"), thresholds.get("confidence")

    def get_stats(self) -> Dict:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing statistics including:
            - total_checked: Total predictions evaluated
            - routed: Predictions sent to LLM
            - blocked_by_entity: Predictions blocked due to entity type
            - blocked_by_threshold: Predictions not uncertain enough
            - routed_by_entity: Breakdown of routed predictions by entity type
            - routing_rate: Percentage of predictions routed
        """
        stats = self.stats.copy()
        if stats["total_checked"] > 0:
            stats["routing_rate"] = (
                f"{100 * stats['routed'] / stats['total_checked']:.2f}%"
            )
        else:
            stats["routing_rate"] = "0%"
        return stats

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.stats = {
            "total_checked": 0,
            "routed": 0,
            "blocked_by_entity": 0,
            "blocked_by_continuation": 0,
            "blocked_by_threshold": 0,
            "routed_by_entity": {},
        }

    def __repr__(self) -> str:
        n_routable = "all" if self.routable_entities is None else f"{len(self.routable_entities)}"
        n_routable_i = "all" if self.routable_i_entities is None else f"{len(self.routable_i_entities)}"
        return (
            f"EntitySpecificRouter("
            f"entropy={self.entropy_threshold}, "
            f"confidence={self.confidence_threshold}, "
            f"selective={self.enable_selective}, "
            f"block_I={self.block_continuation_tokens}, "
            f"routable_B={n_routable} types, "
            f"routable_I={n_routable_i} types, "
            f"blocked={len(self.blocked_entities)} types)"
        )
