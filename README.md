<div align="center">
  <h1>NerGuard</h1>
  <p><strong>Entropy-Gated Hybrid NER for Privacy-Compliant PII Detection</strong></p>

  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-local%20inference-black?style=flat&logo=ollama&logoColor=white" alt="Ollama"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=flat&logo=astral&logoColor=white" alt="uv"></a>
  <a href="https://platform.openai.com/"><img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI"></a>
  <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-integration-1C3C3C?style=flat&logo=langchain&logoColor=white" alt="LangChain"></a>
  <a href="https://pypi.org/project/nerguard/"><img src="https://img.shields.io/badge/PyPI-nerguard-3775A9?style=flat&logo=pypi&logoColor=white" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat" alt="MIT License">
  <br><br>
  <a href="https://huggingface.co/exdsgift/NerGuard-0.3B">🤗 Model on HuggingFace</a>
  <br><br>
</div>
NerGuard is a pre-ingestion privacy layer for RAG pipelines: it detects and redacts PII from text before documents are indexed, keeping sensitive data out of vector databases and LLM context windows. It runs a multilingual mDeBERTa-v3 base model for fast, high-confidence predictions, then selectively routes only uncertain spans to an LLM (OpenAI or local Ollama) for correction, typically less than 3% of tokens. A three-stage regex layer handles structured PII (credit cards, SSNs, IBANs) with deterministic validation. The result is a hybrid pipeline that matches or exceeds larger models on PII recall while remaining GDPR-auditable: every prediction carries its source, confidence score, and routing decision.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2a250234-d7c8-4378-bc06-fd66705ea400" width="800" alt="NerGuard demo">
</div>

### Install

```bash
pip install nerguard
```

The NER model (~300 MB) downloads automatically from HuggingFace on first use.

### Quick start
<a href="https://colab.research.google.com/github/exdsgift/NerGuard/blob/main/scripts/NerGuard_Demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

```python
from nerguard import Redactor

ng = Redactor(
    model_path=None,        # str  — local path or HuggingFace Hub ID for the NER model
    llm_routing=False,      # bool — enable entropy-gated LLM routing
    llm_source="openai",    # str  — "openai" or "ollama"
    llm_model="gpt-4o",     # str  — LLM model name
    api_key=None,           # str  — API key for OpenAI (or None to use OPENAI_API_KEY env var)
    typed=True,             # bool — typed placeholders ([NAME]) vs generic ([PII])
)
result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")

print(result.text)
# "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"

print(result.mapping)
# {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}

print(result.entities)
# [{"label": "GIVENNAME", "text": "John", "confidence": 0.998, "source": "base"}, ...]
```

**Batch:**

```python
texts = [
    "Hi, I'm John Smith. Email: john@acme.com",
    "Call me at +1-800-555-0199 or find me on LinkedIn.",
]

results = [ng.redact(t) for t in texts]  # model stays cached across calls

for r in results:
    print(r.text)
# "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"
# "Call me at [PHONE] or find me on LinkedIn."

# Collect all mappings
all_mappings = {k: v for r in results for k, v in r.mapping.items()}
# {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com", "PHONE_0": "+1-800-555-0199"}
```

## LLM routing

Improves recall on ambiguous spans (phone numbers, IDs, dates) by routing uncertain predictions to an LLM.

```python
# Cloud — pass key explicitly or set OPENAI_API_KEY env var
ng = Redactor(llm_routing=True, llm_source="openai", llm_model="gpt-4o", api_key="sk-...")

# Local — no data leaves the machine (requires Ollama)
ng = Redactor(llm_routing=True, llm_source="ollama", llm_model="qwen2.5:7b")
```

## CLI / interactive REPL

```bash
nerguard                                         # interactive REPL
nerguard --file report.txt                       # redact a file
nerguard --llm --backend ollama --model qwen2.5:7b  # with local LLM
nerguard --format rag                            # RAG-optimised output
```

| REPL command | Description |
|---|---|
| `/mode [human\|rag\|json\|generic]` | Switch output format |
| `/llm` | Toggle LLM routing |
| `/backend [openai\|ollama]` | Switch LLM backend |
| `/model NAME` | Set LLM model |
| `/file PATH` | Redact a file |
| `/help` | Show all commands |

## Constructor parameters

```python
Redactor(
    model_path=None,        # str  — local path or HuggingFace Hub ID for the NER model
    llm_routing=False,      # bool — enable entropy-gated LLM routing
    llm_source="openai",    # str  — "openai" or "ollama"
    llm_model="gpt-4o",     # str  — LLM model name
    api_key=None,           # str  — API key for OpenAI (or None to use OPENAI_API_KEY env var)
    typed=True,             # bool — typed placeholders ([NAME]) vs generic ([PII])
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | HuggingFace auto-download | Local filesystem path or HuggingFace Hub ID for the NER model. Omit to download `exdsgift/NerGuard-0.3B` automatically on first use. |
| `llm_routing` | `bool` | `False` | Enable entropy-gated LLM routing. When `True`, spans where the base model is uncertain are re-evaluated by the LLM. Improves recall on ambiguous tokens (phone numbers, dates, IDs) at the cost of extra latency. |
| `llm_source` | `str` | `"openai"` | LLM backend to use when `llm_routing=True`. `"openai"` calls the OpenAI API; `"ollama"` runs inference locally via Ollama (no data leaves the machine). |
| `llm_model` | `str` | `"gpt-4o"` | Model name passed to the selected LLM backend. Examples: `"gpt-4o"`, `"gpt-4o-mini"` for OpenAI; `"qwen2.5:7b"`, `"llama3.1:8b"` for Ollama. Only used when `llm_routing=True`. |
| `api_key` | `str` | `None` | API key for the OpenAI backend. If `None`, falls back to the `OPENAI_API_KEY` environment variable. Ignored when `llm_source="ollama"`. |
| `typed` | `bool` | `True` | Controls placeholder style. `True` → typed placeholders such as `[NAME]`, `[EMAIL]`, `[PHONE]` (preserves semantic context for downstream LLMs). `False` → every entity becomes `[PII]` regardless of type (maximum compression, no semantic signal). |

## RedactResult fields

`ng.redact(text)` returns a `RedactResult` dataclass with three fields:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Redacted text with placeholders replacing PII spans. |
| `entities` | `list[dict]` | One dict per detected entity, with keys: `label` (entity type), `text` (original value), `start`/`end` (char offsets), `confidence` (0–1), `source` (`"base"` or `"llm"`). |
| `mapping` | `dict[str, str]` | Maps each placeholder instance to its original value, keyed as `"<LABEL>_<index>"` (e.g. `"NAME_0"`, `"EMAIL_0"`). Useful for auditing or selective de-redaction. |

```python
result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")

result.text
# "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"

result.mapping
# {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}

result.entities
# [
#   {"label": "GIVENNAME", "text": "John",          "start": 8,  "end": 12, "confidence": 0.998, "source": "base"},
#   {"label": "SURNAME",   "text": "Smith",         "start": 13, "end": 18, "confidence": 0.995, "source": "base"},
#   {"label": "EMAIL",     "text": "john@acme.com", "start": 27, "end": 40, "confidence": 0.991, "source": "base"},
# ]
```

## Detected entity types

`GIVENNAME` · `SURNAME` · `EMAIL` · `TELEPHONENUM` · `SOCIALNUM` · `CREDITCARDNUMBER` · `IBAN` · `PASSPORTNUM` · `IDCARDNUM` · `DRIVERLICENSENUM` · `TAXNUM` · `STREET` · `BUILDINGNUM` · `CITY` · `ZIPCODE` · `DATE` · `TIME` · `AGE` · `SEX` · `TITLE`

## LangChain integration

NerGuard works as a LangChain **DocumentTransformer** and **Tool** out of the box.

```bash
pip install nerguard[langchain]
```

**Anonymize documents in a RAG pipeline:**

```python
from langchain_core.documents import Document
from nerguard.langchain import NerGuardAnonymizer

anonymizer = NerGuardAnonymizer()
docs = [Document(page_content="John Smith's email is john@acme.com")]
anon_docs = anonymizer.transform_documents(docs)

print(anon_docs[0].page_content)
# "John Smith's email is [EMAIL]"

print(anon_docs[0].metadata["nerguard_mapping"])
# {"EMAIL_0": "john@acme.com"}
```

**As a Tool for LangChain agents:**

```python
from nerguard.langchain import NerGuardTool

tool = NerGuardTool()
result = tool.invoke({"text": "Call Alice at +33 6 12 34 56 78"})
# "Call [NAME] at [PHONE]"
```

## Links

- [Model on HuggingFace](https://huggingface.co/exdsgift/NerGuard-0.3B)
- [GitHub](https://github.com/exdsgift/NerGuard)

## License

MIT
