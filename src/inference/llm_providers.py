"""
Concrete LLM provider implementations for NerGuard.

Provides OpenAI and Ollama backends extracted from LLMRouter.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.core.base_llm import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI, AsyncOpenAI

        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_format or {"type": "json_object"},
            max_tokens=max_tokens,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI returned non-JSON response: {e}") from e

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_format or {"type": "json_object"},
            max_tokens=max_tokens,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI returned non-JSON response: {e}") from e


class OllamaProvider(LLMProvider):
    """Ollama local API provider."""

    def __init__(self, model: str = "qwen2.5:3b"):
        try:
            import ollama
            self._ollama = ollama
        except ImportError:
            raise ImportError("ollama package required. Install with: pip install ollama")
        self.model = model

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 150,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": -1},
        )
        msg = response["message"]
        raw_content = msg.content
        thinking = getattr(msg, "thinking", None) or ""

        # Ollama thinking models may return content as a list of blocks
        if isinstance(raw_content, list):
            raw_content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_content
            )

        # Extract the first valid JSON object from content or thinking field
        decoder = json.JSONDecoder()
        for source in (str(raw_content), str(thinking)):
            content = source.strip()
            idx = 0
            while idx < len(content):
                try:
                    obj, _ = decoder.raw_decode(content, idx)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                next_brace = content.find("{", idx + 1)
                if next_brace == -1:
                    break
                idx = next_brace

        raise ValueError("No JSON found in model response")
