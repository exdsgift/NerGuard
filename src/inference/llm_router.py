"""
LLM Router for NerGuard.

This module provides intelligent routing to LLM for disambiguating
uncertain NER predictions, with caching and robust error handling.
"""

import hashlib
import json
import logging
import os
from typing import Dict, Any, Optional, Set

from src.core.constants import (
    DEFAULT_LLM_SOURCE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_CACHE_SIZE,
    VALID_LABELS_SET,
    ROUTABLE_ENTITIES,
    BLOCKED_ENTITIES,
)
from src.inference.prompts import PROMPT, VALID_LABELS_STR

logger = logging.getLogger(__name__)


class LLMCache:
    """
    In-memory cache for LLM responses.

    Uses MD5 hash of the input context as the cache key.
    Implements FIFO eviction when max_size is reached.

    Attributes:
        max_size: Maximum number of entries to cache
        hits: Number of cache hits
        misses: Number of cache misses
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(
        self,
        target: str,
        context: str,
        prev_label: str,
        current_pred: str,
    ) -> str:
        """Create a unique hash key for the input."""
        content = f"{target}|{context}|{prev_label}|{current_pred}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self,
        target: str,
        context: str,
        prev_label: str,
        current_pred: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached result.

        Args:
            target: Target token
            context: Context string
            prev_label: Previous token's label
            current_pred: Current model prediction

        Returns:
            Cached result if found, None otherwise
        """
        key = self._make_key(target, context, prev_label, current_pred)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(
        self,
        target: str,
        context: str,
        prev_label: str,
        current_pred: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Store a result in the cache.

        Args:
            target: Target token
            context: Context string
            prev_label: Previous token's label
            current_pred: Current model prediction
            result: LLM result to cache
        """
        # FIFO eviction if at capacity
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self._make_key(target, context, prev_label, current_pred)
        self.cache[key] = result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self.cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class LLMRouter:
    """
    Router for LLM-based disambiguation of uncertain NER predictions.

    Supports both OpenAI and Ollama backends with caching and robust error handling.

    Example:
        >>> router = LLMRouter(source="openai")
        >>> result = router.disambiguate(
        ...     target_token="John",
        ...     full_text="Dear John, thank you for your order.",
        ...     char_start=5,
        ...     char_end=9,
        ...     current_pred="O",
        ...     prev_label="O",
        ... )
        >>> print(result["corrected_label"])
        B-GIVENNAME
    """

    def __init__(
        self,
        source: str = DEFAULT_LLM_SOURCE,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE,
        valid_labels: Optional[Set[str]] = None,
    ):
        """
        Initialize the LLM Router.

        Args:
            source: LLM backend ("openai" or "ollama")
            api_key: OpenAI API key (optional, reads from env if not provided)
            model: OpenAI model name
            ollama_model: Ollama model name
            enable_cache: Whether to enable response caching
            cache_size: Maximum cache size
            valid_labels: Set of valid label strings for validation
        """
        self.source = source.lower()
        self.model = model if self.source == "openai" else ollama_model
        self.cache = LLMCache(max_size=cache_size) if enable_cache else None
        self.valid_labels = valid_labels or VALID_LABELS_SET
        self.client = None

        # Initialize client
        if self.source == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                logger.info(f"[LLM] Backend: OpenAI ({self.model})")
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        elif self.source == "ollama":
            try:
                import ollama
                self._ollama = ollama
                logger.info(f"[LLM] Backend: Ollama ({self.model})")
            except ImportError:
                raise ImportError("ollama package required. Install with: pip install ollama")

        else:
            raise ValueError(f"Invalid LLM source: {source}. Use 'openai' or 'ollama'")

    def disambiguate(
        self,
        target_token: str,
        full_text: str,
        char_start: int,
        char_end: int,
        current_pred: str,
        prev_label: str,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Disambiguate a token using LLM.

        Args:
            target_token: The token to classify
            full_text: Full text containing the token
            char_start: Character start position of target
            char_end: Character end position of target
            current_pred: Current model prediction
            prev_label: Previous token's label
            lang: Language code (for context)

        Returns:
            Dictionary with:
                - is_pii: Whether the token is PII
                - corrected_label: The corrected label
                - reasoning: LLM's reasoning
                - cached: Whether result was from cache
        """
        # Extract context
        context = self._extract_context(full_text, char_start, char_end)
        clean_token = full_text[char_start:char_end].strip()

        # Check cache
        if self.cache:
            cached_result = self.cache.get(clean_token, context, prev_label, current_pred)
            if cached_result:
                cached_result = cached_result.copy()
                cached_result["cached"] = True
                return cached_result

        # Format prompt
        try:
            prompt = PROMPT.format(
                context=context,
                target_token=clean_token,
                prev_label=prev_label,
                current_pred=current_pred,
                valid_labels_str=VALID_LABELS_STR,
            )
        except KeyError as e:
            return self._error_response(current_pred, f"Prompt formatting error: {e}")

        # Call LLM
        try:
            raw_result = self._call_llm(prompt)
            validated_result = self._validate_response(raw_result, current_pred, prev_label)

            # Cache result
            if self.cache:
                self.cache.set(clean_token, context, prev_label, current_pred, validated_result)

            validated_result["cached"] = False
            return validated_result

        except Exception as e:
            logger.error(f"[LLM ERROR]: {e}")
            return self._error_response(current_pred, str(e))

    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 400,
    ) -> str:
        """Extract context window around the target token."""
        # Left context with word boundary snapping
        ctx_start = max(0, start - window)
        if ctx_start > 0:
            while ctx_start > 0 and text[ctx_start] not in " \n.":
                ctx_start -= 1
            ctx_start += 1

        # Right context with word boundary snapping
        ctx_end = min(len(text), end + window)
        if ctx_end < len(text):
            while ctx_end < len(text) and text[ctx_end] not in " \n.":
                ctx_end += 1

        prefix = text[ctx_start:start].replace("\n", " ")
        target = text[start:end]
        suffix = text[end:ctx_end].replace("\n", " ")

        return f"...{prefix}>>> {target} <<<{suffix}..."

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Route to appropriate LLM backend."""
        if self.source == "openai":
            return self._call_openai(prompt)
        else:
            return self._call_ollama(prompt)

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a PII classification expert. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        return json.loads(response.choices[0].message.content)

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API."""
        response = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "num_predict": 150},
        )
        return json.loads(response["message"]["content"])

    def _validate_bio_consistency(
        self,
        prev_label: str,
        corrected_label: str,
    ) -> bool:
        """
        Validate BIO consistency of LLM correction.

        Returns False if the correction violates BIO rules, indicating
        we should reject it and keep the model's original prediction.
        """
        # Rule 1: I- cannot follow O
        if corrected_label.startswith("I-") and prev_label == "O":
            logger.debug(f"BIO violation: I- tag '{corrected_label}' cannot follow O")
            return False

        # Rule 2: I- must match previous entity type
        if corrected_label.startswith("I-"):
            prev_type = prev_label.replace("B-", "").replace("I-", "")
            curr_type = corrected_label.replace("I-", "")
            if prev_type != curr_type:
                logger.debug(
                    f"BIO violation: I-{curr_type} cannot follow {prev_label} (type mismatch)"
                )
                return False

        return True

    def _validate_response(
        self,
        raw: Dict[str, Any],
        fallback_label: str,
        prev_label: str = "O",
    ) -> Dict[str, Any]:
        """Validate and normalize LLM response."""
        # Support both V6 format ("label") and older formats ("corrected_label")
        label = raw.get("label") or raw.get("corrected_label", fallback_label)
        label = label.strip().upper()
        reasoning = raw.get("reasoning", "")[:200]

        # Validate label is in valid set
        if label not in self.valid_labels:
            logger.warning(f"[WARNING] Invalid label '{label}' → fallback to '{fallback_label}'")
            label = fallback_label

        # Validate BIO consistency - reject corrections that violate BIO rules
        if not self._validate_bio_consistency(prev_label, label):
            logger.info(f"[BIO REJECT] LLM correction '{label}' violates BIO rules → keeping '{fallback_label}'")
            label = fallback_label
            reasoning = f"BIO violation rejected: {reasoning}"

        return {
            "is_pii": label != "O",
            "corrected_label": label,
            "reasoning": reasoning,
        }

    def _error_response(self, fallback_label: str, error_msg: str) -> Dict[str, Any]:
        """Generate fallback response on error."""
        return {
            "is_pii": False,
            "corrected_label": fallback_label,
            "reasoning": f"Error: {error_msg}",
            "cached": False,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        return self.cache.get_stats() if self.cache else None

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()

    @staticmethod
    def should_route(
        current_pred: str,
        entropy: float,
        confidence: float,
        entropy_threshold: float,
        confidence_threshold: float,
        use_selective_routing: bool = True,
    ) -> bool:
        """
        Determine if a token should be routed to LLM based on entity type and uncertainty.

        This implements selective entity routing: only route entities where LLM
        has proven beneficial (numeric patterns), block entities where LLM causes harm.

        Args:
            current_pred: Model's current prediction (e.g., "B-SURNAME")
            entropy: Model's entropy for this prediction
            confidence: Model's confidence for this prediction
            entropy_threshold: Entropy threshold for uncertainty
            confidence_threshold: Confidence threshold for uncertainty
            use_selective_routing: Whether to apply entity-type filtering

        Returns:
            True if the token should be routed to LLM, False otherwise
        """
        # Basic uncertainty check
        is_uncertain = entropy > entropy_threshold and confidence < confidence_threshold

        if not is_uncertain:
            return False

        if not use_selective_routing:
            return True

        # Extract entity type (remove B-/I- prefix)
        entity_type = current_pred.replace("B-", "").replace("I-", "")

        # Block routing for entities where LLM causes harm
        if entity_type in BLOCKED_ENTITIES:
            logger.debug(f"Routing blocked for entity type: {entity_type}")
            return False

        # Allow routing for "O" predictions (potential false negatives) and routable entities
        if entity_type == "O" or entity_type in ROUTABLE_ENTITIES:
            return True

        # Default: don't route unknown entity types
        return False
