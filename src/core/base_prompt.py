"""
Abstract prompt provider for NerGuard.

PromptProviders supply task-specific prompts and system messages for LLM
disambiguation. For PII this includes bias toward recall ("when uncertain,
prefer PII over O"); for biomedical NER the bias might differ.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set


class PromptProvider(ABC):
    """Abstract base for task-specific LLM prompt generation.

    Subclasses define:
    1. The system message for the LLM (role/persona).
    2. The span disambiguation prompt template.
    3. The O-span recovery prompt template (optional).
    4. The set of valid entity classes for response validation.
    """

    @abstractmethod
    def system_message(self) -> str:
        """Return the system-level message for the LLM (persona/role)."""
        ...

    @abstractmethod
    def span_prompt(
        self,
        context: str,
        span_text: str,
        token_count: int,
        entity_class: str,
        entity_classes_str: str,
    ) -> str:
        """Format the span disambiguation prompt.

        Args:
            context: Text window around the target span (with >>> <<< markers).
            span_text: The actual span text.
            token_count: Number of tokens in the span.
            entity_class: Model's predicted entity class (without BIO prefix).
            entity_classes_str: JSON string of valid entity classes.
        """
        ...

    def o_span_prompt(
        self,
        context: str,
        span_text: str,
        token_count: int,
        target_labels_str: str,
    ) -> Optional[str]:
        """Format the O-span recovery prompt (optional).

        Returns None if this task does not support O-span recovery routing.
        """
        return None

    @abstractmethod
    def valid_entity_classes(self) -> Set[str]:
        """Return the set of valid entity class names (without BIO prefix) plus 'O'."""
        ...

    def entity_class_aliases(self) -> Dict[str, str]:
        """Return a mapping of alias -> canonical entity class name.

        Used to normalize LLM responses that use dataset-specific names.
        Example: {"ssn": "SOCIALNUM", "first_name": "GIVENNAME"}
        """
        return {}
