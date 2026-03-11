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

NerGuard acts as a pre-ingestion privacy layer for RAG pipelines, automatically detecting and redacting PII from documents before they are chunked, embedded, and stored in vector databases. This ensures that sensitive personal data never reaches the retrieval index, keeping downstream LLM queries compliant with GDPR and similar regulations, without sacrificing retrieval quality or requiring manual review.

### 🔍 Quick Example

```text
Input:
  "Hi, I'm John Smith. My SSN is 078-05-1120 and my credit card
   is 4532-0151-1283-0366. Reach me at john@acme.com or +1 555-123-4567."

Detected PII:
  FIRSTNAME          → "John"                    [base model,        conf: 0.998]
  LASTNAME           → "Smith"                   [base model,        conf: 0.997]
  SSN                → "078-05-1120"             [base + regex,      conf: 0.921]
  CREDITCARDNUMBER   → "4532-0151-1283-0366"     [regex override,    conf: 0.999]
  EMAIL              → "john@acme.com"           [base model,        conf: 0.995]
  PHONENUMBER        → "+1 555-123-4567"         [llm routed,        conf: 0.878]

Standard output (--format human):
  "Hi, I'm █████ █████. My SSN is █████ and my credit card
   is █████. Reach me at █████ or █████."

RAG output (--format rag):
  "Hi, I'm [NAME] [NAME]. My SSN is [SSN] and my credit card
   is [CC]. Reach me at [EMAIL] or [PHONE]."
```

Each prediction carries full provenance: base confidence, entropy score, routing decision, and regex validation outcome; enabling auditability for GDPR Data Protection Impact Assessments (DPIA).

### 🏠 Local LLM Backends

NerGuard is backend-agnostic. `qwen2.5:7b` is the recommended local backend: near-identical quality, zero API cost, ~5 GB VRAM. Start Ollama and pass `--backend ollama --model qwen2.5:7b`.

### ⚙️ How It Works

**1. 🧠 Entropy-gated routing**: The base model's per-token softmax distribution is evaluated at inference time. Spans where Shannon entropy exceeds a calibrated threshold (or confidence falls below it) are flagged as uncertain. Only those spans (~3% of tokens in practice) are forwarded to the LLM, preserving the base model's confident predictions and minimizing cost.

**2. 📐 Span-level anchor propagation**: The routing decision is made on the B-token (entity head) and propagated to all I-tokens in the span. This eliminates the per-token oscillation problem: without anchoring, ~75% of LLM-induced errors come from I-tokens being classified differently than their B-token. One LLM call per entity span, not per token.

**3. ✅ Three-mode regex validation**: A structured post-processing layer operates at three pipeline stages: *pre-scan* (Luhn check force-overrides credit card predictions before neural inference), *demotion* (invalidates predictions that fail format validation, e.g. malformed SSNs), and *post-processing* (promotes regex-confirmed patterns the model missed entirely).

Each prediction is tagged with its source (`base`, `llm_routed`, `base+regex`, `regex_override`) for full auditability.

## 🚀 Getting Started

```bash
git clone https://github.com/exdsgift/NerGuard.git
cd NerGuard
./setup.sh
```

`setup.sh` installs all dependencies, optionally configures your OpenAI API key, and reminds you about Ollama for local inference. The NER model (~300 MB) downloads automatically from HuggingFace on first run.

### Install

```bash
pip install nerguard
```

### CLI

```bash
nerguard "Hi, I'm John Smith. Email: john@acme.com"   # default: █████ blocks
nerguard "..." --rag                                   # typed placeholders [NAME] [EMAIL]
nerguard "..." --rag --mapping                         # also show entity→value map
nerguard "..." --json                                  # machine-readable JSON
nerguard "..." --generic                               # compact [PII], max token savings
echo "John Smith, john@acme.com" | nerguard --rag      # stdin support
nerguard -f report.txt --rag

# LLM routing (optional, improves recall on ambiguous spans)
nerguard "..." --llm --backend ollama --model qwen2.5:7b
nerguard "..." --llm --backend openai --model gpt-4o-mini

nerguard --help
```

| Flag | Output | Use case |
| --- | --- | --- |
| *(none)* | `█████` blocks, colored | Human review, auditing |
| `--rag` | `[NAME]` `[EMAIL]` `[SSN]`… | RAG / LLM pipeline input |
| `--json` | Machine-readable JSON | API integration, logging |
| `--generic` | `[PII]` only | Maximum token compression |

Shell shortcuts (for repo users):

```bash
./redact.sh "..."      # → nerguard "..."
./rag_redact.sh "..."  # → nerguard "..." --rag
```

### Python API (RAG)

```python
from src.rag import nerguard

ng = nerguard()
result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")

print(result.text)
# "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"

print(result.mapping)
# {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}

# Batch processing
results = ng.redact_batch(["doc 1...", "doc 2...", "doc 3..."])
```

### Python API (base)

```python
from src.inference.tester import PIITester

tester = PIITester(model_path="exdsgift/NerGuard-0.3B")
entities = tester.get_entities("John Smith lives at 123 Main St. Email: john@email.com")

for e in entities:
    print(f"{e['label']}: {e['text']} (conf: {e['confidence']:.2%}, source: {e['source']})")
```

### Reproducing the Benchmark

```bash
# Full cross-system benchmark (cloud)
uv run python -m src.benchmark.runner \
  --systems nerguard-hybrid-v2,nerguard-hybrid,nerguard-base,presidio,spacy,piiranha,bert-ner \
  --datasets nvidia-pii --samples 1000 --llm-model gpt-4o --batch-llm \
  --semantic-alignment alignments/default.json

# Local inference — no data leaves the machine
uv run python -m src.benchmark.runner \
  --systems nerguard-hybrid-v2 --datasets nvidia-pii --samples 1000 \
  --llm-source ollama --llm-model qwen2.5:7b --batch-llm \
  --semantic-alignment alignments/default.json
```

---

## 📁 Repository Structure

```text
nerguard_rag/      nerguard-rag package — RAG Python API (pip install nerguard-rag)
src/
  core/            Route config, base abstractions (ValidationStrategy, PromptProvider)
  inference/       LLM router, entity router, regex validator, span assembler
  tasks/           Task plugins: pii/, biomedical/, financial/
  training/        Model training and validation
  benchmark/       Cross-system benchmark framework (runner, metrics, datasets, systems)
  optimization/    Threshold calibrator, ONNX quantization
  scripts/         CLI entry points (nerguard) and analysis runners
docs/              Technical notes, architecture diagrams, bibliography
experiments/       Benchmark results (JSON + summaries)
alignments/        Semantic label alignment for cross-system evaluation
```

---

## 📖 Citation

```bibtex
@mastersthesis{durante2026nerguard,
  title     = {Engineering a Scalable Multilingual PII Detection System
               with mDeBERTa-v3 and LLM-Based Validation},
  author    = {Durante, Gabriele},
  year      = {2026},
  school    = {University of Verona},
  department = {Department of Computer Science}
}
```

## License

MIT
