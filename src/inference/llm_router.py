"""
LLM Router for NerGuard.

This module provides intelligent routing to LLM for disambiguating
uncertain NER predictions, with caching and robust error handling.

The router is task-agnostic when used with a PromptProvider: the provider
supplies system messages, prompt templates, valid labels, and entity
aliases. Without a PromptProvider, it defaults to PII behavior for
backward compatibility.
"""

import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Dict, Any, Optional, Set

from src.core.base_llm import LLMProvider

from src.core.constants import (
    DEFAULT_LLM_SOURCE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_CACHE_SIZE,
    VALID_LABELS_SET,
    ROUTABLE_ENTITIES,
    BLOCKED_ENTITIES,
    ENTITY_CLASSES_WITH_O,
    EXTENDED_ENTITY_CLASSES_WITH_O,
    NVIDIA_CLASS_TO_BASE,
)
from src.inference.prompts import PROMPT, PROMPT_V12, PROMPT_V13, PROMPT_V14_SPAN, ENTITY_CLASSES_STR, VALID_LABELS_STR

if TYPE_CHECKING:
    from src.core.base_prompt import PromptProvider

logger = logging.getLogger(__name__)


def _safe_str(s: object) -> str:
    """Escape curly braces in user-supplied strings before .format() interpolation.

    Prevents both KeyError crashes and prompt injection via malformed braces.
    """
    return str(s).replace("{", "{{").replace("}", "}}")


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
        context_window: int = 400,
        span_prompt_version: str = "V14_SPAN",
        use_structured_outputs: bool = True,
        use_extended_labels: bool = False,
        prompt_provider: Optional["PromptProvider"] = None,
        provider: Optional[LLMProvider] = None,
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
            context_window: Character window size around target for context extraction
            span_prompt_version: Prompt version for span routing (V14_SPAN, V15_SPAN, V16_SPAN)
            use_structured_outputs: Use OpenAI json_schema mode (only for OpenAI)
            use_extended_labels: Accept NVIDIA-alias entity names in responses (e.g. "ssn",
                "certificate_license_number"); mapped to base model classes automatically.
                Automatically enabled when span_prompt_version=="V16_SPAN".
            prompt_provider: Task-specific prompt provider. When set, overrides
                system_message, valid_entity_classes, and entity aliases.
                When None, defaults to PII behavior.
            provider: Pre-built LLMProvider instance. When given, source/api_key/model
                are ignored for client creation (backward-compatible).
        """
        self.source = source.lower()
        self.model = model if self.source == "openai" else ollama_model
        self.cache = LLMCache(max_size=cache_size) if enable_cache else None
        self.valid_labels = valid_labels or VALID_LABELS_SET
        self.prompt_provider = prompt_provider
        self.context_window = context_window
        self.use_structured_outputs = use_structured_outputs and self.source == "openai"

        if prompt_provider is not None:
            # Task-agnostic path: use provider for labels, aliases, system message
            self.valid_entity_classes = prompt_provider.valid_entity_classes()
            self._entity_aliases = prompt_provider.entity_class_aliases()
            self._system_message = prompt_provider.system_message()
        else:
            # Legacy PII path: use hardcoded constants
            _use_extended = use_extended_labels or (span_prompt_version == "V16_SPAN")
            self.valid_entity_classes = EXTENDED_ENTITY_CLASSES_WITH_O if _use_extended else ENTITY_CLASSES_WITH_O
            self._entity_aliases = NVIDIA_CLASS_TO_BASE
            self._system_message = "You are a PII classification expert. Output only valid JSON."

        # V9 for OpenAI (JSON mode, full BIO labels)
        # V13 for Ollama (class-only paradigm: LLM predicts class, BIO assigned deterministically)
        self.prompt_template = PROMPT if self.source == "openai" else PROMPT_V13
        # Span prompt: configurable version for span-level routing
        from src.inference.prompts import PROMPTS
        self.span_prompt_template = PROMPTS.get(span_prompt_version, PROMPT_V14_SPAN)

        # Initialize LLM provider
        if provider is not None:
            self.provider = provider
        elif self.source == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
            from src.inference.llm_providers import OpenAIProvider
            self.provider = OpenAIProvider(api_key=api_key, model=self.model)
            logger.info(f"[LLM] Backend: OpenAI ({self.model})")
        elif self.source == "ollama":
            from src.inference.llm_providers import OllamaProvider
            self.provider = OllamaProvider(model=self.model)
            logger.info(f"[LLM] Backend: Ollama ({self.model})")
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
        # V9 (OpenAI): full BIO labels, uses valid_labels_str
        # V13 (Ollama): class-only, uses entity_classes_str; BIO assigned deterministically after
        try:
            prompt = self.prompt_template.format(
                context=_safe_str(context),
                target_token=_safe_str(clean_token),
                prev_label=_safe_str(prev_label),
                current_pred=_safe_str(current_pred),
                valid_labels_str=VALID_LABELS_STR,
                entity_classes_str=ENTITY_CLASSES_STR,
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

    def disambiguate_span(
        self,
        span_text: str,
        token_count: int,
        full_text: str,
        span_start: int,
        span_end: int,
        current_pred: str,
        prev_label: str,
    ) -> Dict[str, Any]:
        """
        Route an entire entity span to the LLM as a single call (V14_SPAN).

        Instead of routing each token independently, the whole span (e.g. "John Smith")
        is presented to the LLM with its full context. BIO is applied deterministically:
        - corrected_label = B-{class} for the first token (caller applies I-{class} to rest)
        - corrected_label = "O" → caller sets all span tokens to O

        Args:
            span_text: Full span text, e.g. "John Smith"
            token_count: Number of tokens in the span
            full_text: Complete document text (for context extraction)
            span_start: Character start of the first span token
            span_end: Character end of the last span token
            current_pred: Entity class without BIO prefix, e.g. "SURNAME"
            prev_label: BIO label of the token immediately before the span

        Returns:
            {is_pii, corrected_label, reasoning, cached}
        """
        context = self._extract_context(full_text, span_start, span_end)
        clean_span = full_text[span_start:span_end].strip()
        fallback_label = f"B-{current_pred}"

        if self.cache:
            cached = self.cache.get(clean_span, context, prev_label, current_pred)
            if cached:
                result = cached.copy()
                result["cached"] = True
                return result

        try:
            prompt = self.span_prompt_template.format(
                context=_safe_str(context),
                span_text=_safe_str(clean_span),
                token_count=token_count,
                entity_class=_safe_str(current_pred),
                entity_classes_str=ENTITY_CLASSES_STR,
            )
        except KeyError as e:
            return self._error_response(fallback_label, f"Prompt formatting error: {e}")

        try:
            raw_result = self._call_llm(prompt)
            validated = self._validate_response(raw_result, fallback_label, prev_label)

            if self.cache:
                self.cache.set(clean_span, context, prev_label, current_pred, validated)

            validated["cached"] = False
            return validated

        except Exception as e:
            logger.error(f"[LLM SPAN ERROR]: {e}")
            return self._error_response(fallback_label, str(e))

    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window: Optional[int] = None,
    ) -> str:
        """Extract context window around the target token."""
        window = window or self.context_window
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

    def _build_response_format(self) -> Optional[Dict[str, Any]]:
        """Build the response_format dict for the current configuration."""
        if self.use_structured_outputs:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "ner_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "entity_class": {"type": "string"},
                        },
                        "required": ["reasoning", "entity_class"],
                        "additionalProperties": False,
                    },
                },
            }
        return None

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM via the provider interface."""
        messages = [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": prompt},
        ]
        return self.provider.call(
            messages=messages,
            temperature=0.0,
            max_tokens=150,
            response_format=self._build_response_format(),
        )

    @staticmethod
    def _class_to_bio(entity_class: str, prev_label: str) -> str:
        """
        Convert an entity class to a BIO label deterministically.

        Rules:
          - "O" → "O"
          - If prev token was same entity type (B-X or I-X) → I-{class}
          - Otherwise → B-{class}

        This guarantees BIO consistency by construction.
        """
        if entity_class == "O":
            return "O"
        prev_type = prev_label.replace("B-", "").replace("I-", "") if prev_label != "O" else ""
        return f"I-{entity_class}" if prev_type == entity_class else f"B-{entity_class}"

    def _validate_bio_consistency(
        self,
        prev_label: str,
        corrected_label: str,
    ) -> bool:
        """
        Validate BIO consistency of LLM correction.

        Returns False if the correction violates BIO rules, indicating
        we should reject it and keep the model's original prediction.
        Used only for V9/V12 (full BIO label) responses.
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
        """Validate and normalize LLM response.

        Supports two paradigms:
        - V13 (entity_class key): LLM returns class only, BIO assigned via _class_to_bio()
        - V9/V12 (label key): LLM returns full BIO label, validated with BIO consistency check
        """
        reasoning = raw.get("reasoning", "")[:200]

        # --- V13 paradigm: entity class → deterministic BIO ---
        entity_class = raw.get("entity_class")
        if entity_class is not None:
            entity_class = str(entity_class).strip().upper()
            # Strip accidental BIO prefix (V9 prompt + structured output may produce "B-SURNAME")
            if entity_class.startswith(("B-", "I-")):
                entity_class = entity_class[2:]
            if entity_class not in self.valid_entity_classes:
                # Try alias lookup (case-insensitive) before falling back.
                # e.g. "CERTIFICATE_LICENSE_NUMBER" → "DRIVERLICENSENUM"
                alias_mapped = self._entity_aliases.get(entity_class.lower())
                if alias_mapped is not None:
                    logger.debug(f"[alias] '{entity_class}' → '{alias_mapped}'")
                    entity_class = alias_mapped
                else:
                    logger.warning(f"[WARNING] Invalid entity class '{entity_class}' → fallback to '{fallback_label}'")
                    label = fallback_label
                    return {"is_pii": label != "O", "corrected_label": label, "reasoning": reasoning}
            label = self._class_to_bio(entity_class, prev_label)
            logger.debug(f"[V13] class='{entity_class}' prev='{prev_label}' → '{label}'")
            return {"is_pii": label != "O", "corrected_label": label, "reasoning": reasoning}

        # --- V9/V12 paradigm: full BIO label with consistency validation ---
        label = raw.get("label") or raw.get("corrected_label", fallback_label)
        # Some models (e.g. thinking models) return a list of labels; take the first valid one
        if isinstance(label, list):
            valid = [l for l in label if isinstance(l, str) and l.strip().upper() in self.valid_labels]
            label = valid[0] if valid else fallback_label
        label = str(label).strip().upper()

        if label not in self.valid_labels:
            logger.warning(f"[WARNING] Invalid label '{label}' → fallback to '{fallback_label}'")
            label = fallback_label

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

    async def _call_llm_async(self, prompt: str) -> Dict[str, Any]:
        """Call LLM asynchronously via the provider interface."""
        messages = [
            {"role": "system", "content": self._system_message},
            {"role": "user", "content": prompt},
        ]
        return await self.provider.call_async(
            messages=messages,
            temperature=0.0,
            max_tokens=150,
            response_format=self._build_response_format(),
        )

    async def disambiguate_span_async(
        self,
        span_text: str,
        token_count: int,
        full_text: str,
        span_start: int,
        span_end: int,
        current_pred: str,
        prev_label: str,
    ) -> Dict[str, Any]:
        """Async version of disambiguate_span for concurrent API calls."""
        context = self._extract_context(full_text, span_start, span_end)
        clean_span = full_text[span_start:span_end].strip()
        fallback_label = f"B-{current_pred}"

        if self.cache:
            cached = self.cache.get(clean_span, context, prev_label, current_pred)
            if cached:
                result = cached.copy()
                result["cached"] = True
                return result

        try:
            prompt = self.span_prompt_template.format(
                context=_safe_str(context),
                span_text=_safe_str(clean_span),
                token_count=token_count,
                entity_class=_safe_str(current_pred),
                entity_classes_str=ENTITY_CLASSES_STR,
            )
        except KeyError as e:
            return self._error_response(fallback_label, f"Prompt formatting error: {e}")

        try:
            raw_result = await self._call_llm_async(prompt)
            validated = self._validate_response(raw_result, fallback_label, prev_label)

            if self.cache:
                self.cache.set(clean_span, context, prev_label, current_pred, validated)

            validated["cached"] = False
            return validated

        except Exception as e:
            logger.error(f"[LLM ASYNC SPAN ERROR]: {e}")
            return self._error_response(fallback_label, str(e))

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
