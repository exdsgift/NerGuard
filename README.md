<div align="center">
  <h1>NerGuard</h1>
  <p><strong>Entropy-Gated Hybrid NER for Privacy-Compliant PII Detection</strong></p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-local%20inference-black?style=flat&logo=ollama&logoColor=white" alt="Ollama"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=flat&logo=astral&logoColor=white" alt="uv"></a>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat" alt="MIT License">
  <br><br>
  <a href="https://huggingface.co/exdsgift/NerGuard-0.3B">🤗 Model on HuggingFace</a>
  &nbsp;·&nbsp;
  <a href="https://pypi.org/project/nerguard/">📦 PyPI: nerguard</a>
  <br><br>
</div>

NerGuard is a pre-ingestion privacy layer for RAG pipelines: it detects and redacts PII from text before documents are indexed, keeping sensitive data out of vector databases and LLM context windows. It runs a multilingual mDeBERTa-v3 base model for fast, high-confidence predictions, then selectively routes only uncertain spans to an LLM (OpenAI or local Ollama) for correction, typically less than 3% of tokens. A three-stage regex layer handles structured PII (credit cards, SSNs, IBANs) with deterministic validation. The result is a hybrid pipeline that matches or exceeds larger models on PII recall while remaining GDPR-auditable: every prediction carries its source, confidence score, and routing decision.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2a250234-d7c8-4378-bc06-fd66705ea400" width="800" alt="NerGuard demo">
</div>

## Install

```bash
pip install nerguard
```

The NER model (~300 MB) downloads automatically from HuggingFace on first use.

## Quick start

```python
from nerguard import Redactor

ng = Redactor()
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
results = [ng.redact(t) for t in texts]  # model stays cached across calls
```

## LLM routing

Improves recall on ambiguous spans (phone numbers, IDs, dates) by routing uncertain predictions to an LLM.

```python
# Cloud (requires OPENAI_API_KEY)
ng = Redactor(llm_routing=True, llm_source="openai", llm_model="gpt-4o")

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

| Parameter | Default | Description |
|---|---|---|
| `model_path` | HuggingFace auto-download | Local path or Hub ID for the NER model |
| `llm_routing` | `False` | Enable entropy-gated LLM routing |
| `llm_source` | `"openai"` | `"openai"` or `"ollama"` |
| `llm_model` | `"gpt-4o"` | LLM model name |
| `typed` | `True` | `True` → `[NAME]`, `False` → `[PII]` |

## Detected entity types

`GIVENNAME` · `SURNAME` · `EMAIL` · `TELEPHONENUM` · `SOCIALNUM` · `CREDITCARDNUMBER` · `IBAN` · `PASSPORTNUM` · `IDCARDNUM` · `DRIVERLICENSENUM` · `TAXNUM` · `STREET` · `BUILDINGNUM` · `CITY` · `ZIPCODE` · `DATE` · `TIME` · `AGE` · `SEX` · `TITLE`

## Links

- [Model on HuggingFace](https://huggingface.co/exdsgift/NerGuard-0.3B)
- [GitHub](https://github.com/exdsgift/NerGuard)

## License

MIT
