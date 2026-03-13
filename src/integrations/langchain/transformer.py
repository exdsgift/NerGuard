"""NerGuard as a LangChain DocumentTransformer."""

from typing import Any, Optional, Sequence

try:
    from langchain_core.documents import Document
    from langchain_core.documents.transformers import BaseDocumentTransformer
except ImportError as e:
    raise ImportError(
        "langchain-core is required for NerGuard LangChain integration. "
        "Install with: pip install nerguard[langchain]"
    ) from e

from src.rag.redactor import nerguard as Redactor


class NerGuardAnonymizer(BaseDocumentTransformer):
    """Anonymize PII in LangChain Documents using NerGuard.

    Transforms Documents by redacting PII from page_content.
    Entity mappings are stored in document metadata for potential
    de-anonymization.

    Example::

        from nerguard.langchain import NerGuardAnonymizer

        anonymizer = NerGuardAnonymizer()
        docs = anonymizer.transform_documents(documents)
    """

    def __init__(
        self,
        model_path: str = "./models/mdeberta-pii-safe/final",
        llm_routing: bool = False,
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        typed: bool = True,
        api_key: Optional[str] = None,
        metadata_key: str = "nerguard_mapping",
    ) -> None:
        self._redactor = Redactor(
            model_path=model_path,
            llm_routing=llm_routing,
            llm_source=llm_source,
            llm_model=llm_model,
            typed=typed,
            api_key=api_key,
        )
        self._metadata_key = metadata_key

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Redact PII from each document's page_content.

        Args:
            documents: Sequence of LangChain Documents to anonymize.

        Returns:
            New Documents with redacted page_content and mapping in metadata.
        """
        results = []
        for doc in documents:
            result = self._redactor.redact(doc.page_content)
            new_metadata = {**doc.metadata, self._metadata_key: result.mapping}
            results.append(Document(
                page_content=result.text,
                metadata=new_metadata,
            ))
        return results
