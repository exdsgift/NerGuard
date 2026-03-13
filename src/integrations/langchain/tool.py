"""NerGuard as a LangChain Tool for agent use."""

from typing import Optional, Type

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError as e:
    raise ImportError(
        "langchain-core and pydantic are required for NerGuard LangChain integration. "
        "Install with: pip install nerguard[langchain]"
    ) from e

from src.rag.redactor import nerguard as Redactor


class RedactInput(BaseModel):
    """Input schema for the NerGuard redaction tool."""

    text: str = Field(description="The text to redact PII from")


class NerGuardTool(BaseTool):
    """LangChain Tool that redacts PII from text using NerGuard.

    Agents can use this tool to anonymize text before processing it.

    Example::

        from nerguard.langchain import NerGuardTool

        tool = NerGuardTool()
        result = tool.invoke({"text": "Call John at 555-0123"})
    """

    name: str = "nerguard_redact"
    description: str = (
        "Redact personally identifiable information (PII) from text. "
        "Input is raw text, output is the text with PII replaced by "
        "typed placeholders like [NAME], [EMAIL], [PHONE]."
    )
    args_schema: Type[BaseModel] = RedactInput

    _redactor: Redactor = PrivateAttr()
    _last_mapping: dict = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        model_path: str = "./models/mdeberta-pii-safe/final",
        llm_routing: bool = False,
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        typed: bool = True,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._redactor = Redactor(
            model_path=model_path,
            llm_routing=llm_routing,
            llm_source=llm_source,
            llm_model=llm_model,
            typed=typed,
            api_key=api_key,
        )
        self._last_mapping = {}

    def _run(self, text: str) -> str:
        """Redact PII from the input text."""
        result = self._redactor.redact(text)
        self._last_mapping = result.mapping
        return result.text

    @property
    def last_mapping(self) -> dict:
        """Access the entity mapping from the most recent redaction."""
        return self._last_mapping
