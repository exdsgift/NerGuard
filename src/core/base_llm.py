"""
Abstract LLM provider for NerGuard.

LLMProviders encapsulate the API-specific logic for calling an LLM backend
(OpenAI, Ollama, etc.), allowing the LLMRouter to work against a stable
interface regardless of the underlying service.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base for LLM backend providers.

    Subclasses implement synchronous and optionally asynchronous
    call methods for a specific LLM API.
    """

    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send messages to the LLM and return parsed JSON response.

        Args:
            messages: List of {"role": str, "content": str} message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            response_format: Optional JSON schema / format specification.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        ...

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of call(). Default raises NotImplementedError."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support async calls"
        )
