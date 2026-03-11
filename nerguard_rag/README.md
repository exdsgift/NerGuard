# nerguard-rag

RAG-optimized PII redaction with typed placeholders, built on [NerGuard](https://github.com/exdsgift/NerGuard).

Replaces detected PII with compact typed markers (`[NAME]`, `[EMAIL]`, `[SSN]`, …) that signal removed content to downstream LLMs while preserving semantic structure and minimizing token usage.

## Install

```bash
pip install nerguard-rag
```

## Usage

```python
from nerguard_rag import nerguard

ng = nerguard()
result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")

print(result.text)
# "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"

print(result.mapping)
# {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}
```

### Batch processing

```python
results = ng.redact_batch(["doc 1...", "doc 2...", "doc 3..."])
```

### CLI

```bash
nerguard-rag "Hi, I'm John Smith. Email: john@acme.com"
nerguard-rag -f document.txt --mapping
nerguard-rag "..." --format generic   # [PII] instead of typed labels
nerguard-rag "..." --llm --backend ollama --model qwen2.5:7b
```

The model (~300 MB) downloads automatically from [HuggingFace](https://huggingface.co/exdsgift/NerGuard-0.3B) on first run.

## License

MIT
