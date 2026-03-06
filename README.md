<div align="center">
  <h1>NerGuard</h1>
  <p><strong>Entropy-Gated Hybrid NER for Privacy-Compliant PII Detection</strong></p>

  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-compatible-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI"></a>
  <br><br>
  <a href="https://huggingface.co/exdsgift/NerGuard-0.3B">🤗 Model on HuggingFace</a>
</div>

---

Organizations subject to GDPR cannot send documents to external APIs for PII detection, yet purely rule-based tools miss contextual entities, and neural models alone produce too many false negatives to be trusted for compliance. NerGuard resolves this trade-off through a three-stage hybrid architecture: a fine-tuned mDeBERTa-v3-base (279M params) classifies 20 PII entity types across 8 European languages, routing only the spans where it is genuinely uncertain to a secondary LLM for disambiguation,  reducing API calls by ~50% vs. naive always-route strategies. A multi-mode regex layer handles structured patterns (Luhn check for credit cards, format validation for SSNs) before and after neural inference. The system is designed for **fully local, on-premise deployment** via [Ollama](https://ollama.com/): `qwen2.5:7b` achieves within 0.002 F1-macro of GPT-4o at zero API cost, keeping PII entirely within the organization's infrastructure. On the NVIDIA/Nemotron-PII benchmark (1,000 samples, 7 systems), NerGuard ranks **1st on F1-macro (0.5069) and F1-micro (0.7015)** at a median latency of **41 ms** — 2.1× faster than Microsoft Presidio.

---

## 🔍 Quick Example

```text
Input:
  "Hi, I'm John Smith. My SSN is 078-05-1120 and my credit card
   is 4532-0151-1283-0366. Reach me at john@acme.com or +1 555-123-4567."

Detected PII:
  FIRSTNAME          → "John"                  [base model,        conf: 0.998]
  LASTNAME           → "Smith"                 [base model,        conf: 0.997]
  SSN                → "078-05-1120"           [base + regex,      conf: 0.921]
  CREDITCARDNUMBER   → "4532-0151-1283-0366"   [regex override,    conf: 0.999]  ← Luhn check
  EMAIL              → "john@acme.com"          [base model,        conf: 0.995]
  PHONENUMBER        → "+1 555-123-4567"        [llm routed,        conf: 0.878]  ← uncertain span

Redacted output:
  "Hi, I'm [FIRSTNAME] [LASTNAME]. My SSN is [SSN] and my credit card
   is [CREDITCARDNUMBER]. Reach me at [EMAIL] or [PHONENUMBER]."
```

Each prediction carries full provenance: base confidence, entropy score, routing decision, and regex validation outcome; enabling auditability for GDPR Data Protection Impact Assessments (DPIA).

---

## 📊 Benchmark Results

Evaluated on **NVIDIA/Nemotron-PII** (1,000 samples, seed=42). Tier 2 evaluation: semantic alignment over 16 comparable entity types across all systems. Full results and per-entity breakdowns are in [`experiments/`](experiments/).

| System | F1-macro | F1-micro | Entity-F1 | Precision | Recall | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **NerGuard Hybrid V2** (GPT-4o) | **0.5069** | **0.7015** | 0.6634 | 0.5952 | 0.7492 | 41 |
| NerGuard Hybrid V1 (GPT-4o) | 0.4943 | 0.6862 | 0.6475 | 0.5754 | 0.7401 | 31 |
| Presidio | 0.4933 | 0.5493 | **0.6680** | 0.5697 | 0.8073 | 86 |
| Piiranha | 0.4731 | 0.6501 | 0.6195 | **0.6765** | 0.5713 | 31 |
| NerGuard Base (no LLM) | 0.4175 | 0.6105 | 0.6076 | 0.5616 | 0.6619 | 33 |
| spaCy (en_core_web_trf) | 0.3607 | 0.4175 | 0.5527 | 0.4314 | 0.7688 | 144 |
| dslim/bert-base-NER | 0.3331 | 0.4821 | 0.6225 | 0.5409 | 0.7332 | 38 |

The hybrid pipeline provides a **+8.94 pt F1-macro gain** over the base model alone. An ablation across five routing variants confirms that selective routing strictly dominates both never-route and always-route: the full system achieves the same Entity-F1 (0.688) as always-routing with **50% fewer LLM calls** (5,470 vs. 11,024), and outperforms it on F1-macro by +2.46 pts. Statistical significance is confirmed by paired bootstrap (p < 0.001, 95% CI [+0.067, +0.103]) and McNemar's test (χ² = 208.14, p < 0.001) on 500 held-out samples.

### 🏠 Local LLM Backends

NerGuard is backend-agnostic. All eight models tested cluster within **0.030 F1-macro of GPT-4o**, confirming that the base encoder (not the LLM router) is the performance bottleneck.

| Model | Params | F1-macro | Δ vs GPT-4o | Latency (ms) |
| --- | --- | --- | --- | --- |
| GPT-4o (cloud) | — | 0.5069 | — | 41 |
| **qwen2.5:7b** | 7B | **0.5051** | **−0.002** | 564 |
| llama3.1:8b | 8B | 0.4972 | −0.010 | 707 |
| gpt-oss:20b | 20B | 0.5028 | −0.004 | 3,139 |

`qwen2.5:7b` is the recommended local backend: near-identical quality, zero API cost, ~5 GB VRAM. Start Ollama and pass `--llm-source ollama --llm-model qwen2.5:7b` to the benchmark runner.

---

## ⚙️ How It Works

**1. 🧠 Entropy-gated routing**: The base model's per-token softmax distribution is evaluated at inference time. Spans where Shannon entropy exceeds a calibrated threshold (or confidence falls below it) are flagged as uncertain. Only those spans (~3% of tokens in practice) are forwarded to the LLM, preserving the base model's confident predictions and minimizing cost.

**2. 📐 Span-level anchor propagation**: The routing decision is made on the B-token (entity head) and propagated to all I-tokens in the span. This eliminates the per-token oscillation problem: without anchoring, ~75% of LLM-induced errors come from I-tokens being classified differently than their B-token. One LLM call per entity span, not per token.

**3. ✅ Three-mode regex validation**: A structured post-processing layer operates at three pipeline stages: *pre-scan* (Luhn check force-overrides credit card predictions before neural inference), *demotion* (invalidates predictions that fail format validation, e.g. malformed SSNs), and *post-processing* (promotes regex-confirmed patterns the model missed entirely).

Each prediction is tagged with its source (`base`, `llm_routed`, `base+regex`, `regex_override`) for full auditability.

---

## 🌍 Cross-Domain Generalization

The routing pipeline is task-agnostic. The same architecture, with task-specific model, prompt, and auto-calibrated thresholds (`--calibrate N`) — was applied to three additional NER domains:

| Domain | n_classes | Base Entity-F1 | Hybrid Entity-F1 | Δ | p-value |
| --- | --- | --- | --- | --- | --- |
| **PII** (mDeBERTa-v3) | 20 | 0.608 | **0.698** | +0.090 | <0.001 *** |
| **Biomedical** BC5CDR (RoBERTa-large) | 2 | 0.880 | **0.888** | +0.009 | 0.024 * |
| **Financial** BUSTER (BERT-base) | 6 | 0.683 | **0.687** | +0.004 | 0.225 n.s. |
| **Financial** FiNER-139 (DistilBERT) | 139 | **0.664** | 0.554 | −0.110 | negative |

The results reveal a clear **boundary condition**: LLM routing is beneficial when entity classes are few and semantically distinct. Beyond ~13 classes, general-purpose LLMs struggle to disambiguate;  FiNER-139's 139 XBRL tags collapse recall by −23.5 pts as GPT-4o defaults to "O" rather than guessing a fine-grained tag. This is a deliberate honest negative result: the boundary condition is characterized and reported, not suppressed.

Adding a new task requires only a `RouteConfig` (thresholds + entity list), a `PromptProvider` (task-specific LLM prompt), and a dataset adapter. Thresholds auto-calibrate via grid search on held-out samples.

---

## 🚀 Getting Started

```bash
git clone https://github.com/exdsgift/NerGuard.git
cd NerGuard
uv sync
```

### Inference

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

```
src/
  core/            Route config, base abstractions (ValidationStrategy, PromptProvider)
  inference/       LLM router, entity router, regex validator, span assembler
  tasks/           Task plugins: pii/, biomedical/, financial/
  training/        Model training and validation
  benchmark/       Cross-system benchmark framework (runner, metrics, datasets, systems)
  optimization/    Threshold calibrator, ONNX quantization
  scripts/         CLI entry points and analysis runners
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
