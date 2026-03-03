<div align="center">
  <h1>NerGuard</h1>
  <p><strong>GDPR-Compliant PII Detection through Entropy-Based Hybrid NER</strong></p>

  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI"></a>
  <br><br>
  <a href="https://huggingface.co/exdsgift/NerGuard-0.3B">Model on HuggingFace</a>
</div>

---

## Overview

Automated PII detection is a critical requirement for GDPR compliance (Art. 4, Art. 17), yet existing systems face a fundamental trade-off: rule-based approaches (Presidio, regex) achieve high precision on structured entities but miss contextual PII, while transformer-based models generalize better but produce costly false negatives on tail-distribution entities.

**NerGuard** resolves this trade-off through a three-stage hybrid architecture:

1. **Base model**; A fine-tuned mDeBERTa-v3-base (279M params) performs token classification across 20 PII entity types in 8 European languages, producing per-token softmax distributions.
2. **Entropy-gated LLM routing**; Only token spans where the base model exhibits high prediction uncertainty (Shannon entropy > 0.583, confidence < 0.787) are selectively routed to an LLM for disambiguation. Span-level anchor propagation ensures a single LLM call per entity span rather than per token, reducing API costs and eliminating harmful per-token oscillation.
3. **Multi-layer regex validation**; A structured validation pipeline operates in three modes: *pre-scan* (force-override via Luhn check for credit cards), *demotion* (invalidate predictions that fail format validation, e.g. SSN), and *post-processing* (promote regex-confirmed patterns the model missed).

This design achieves **first place on F1-macro (0.5069) and F1-micro (0.7015)** across 7 evaluated systems on the NVIDIA/Nemotron-PII benchmark, while maintaining a median latency of **35.7 ms** per sample ; 2.4x faster than Presidio and 4.0x faster than spaCy.

## Architecture

<p align="center">
  <img src="docs/diagrams/architecture.png" alt="NerGuard Architecture" width="700">
</p>

**Key architectural properties:**
- **Selective routing**; Only uncertain spans reach the LLM, preserving the base model's confident predictions and minimizing API overhead.
- **Anchor propagation**; The B-token's routing decision applies to the entire entity span, eliminating the I-token oscillation problem that causes 75% of LLM-induced errors in per-token routing.
- **Three-mode regex**; Pre-scan, demotion, and post-processing operate at different pipeline stages, each targeting a specific class of errors (false negatives, false positives, and missed patterns respectively).

## Benchmark Results

Evaluated on **NVIDIA/Nemotron-PII** (1,000 samples, seed=42). All systems are compared on the intersection of their label vocabularies via semantic alignment (Tier 2 evaluation, 16 comparable entity types). Full results, per-entity scores, confusion matrices, and error analysis are available in [`experiments/`](experiments/).

### Cross-System Comparison

| System | F1-macro | F1-micro | Entity-F1 | Precision | Recall | Latency (ms) |
|---|---|---|---|---|---|---|
| **NerGuard Hybrid V2** (GPT-4o) | **0.5069** | **0.7015** | 0.6634 | 0.6484 | 0.7641 | 41 |
| NerGuard Hybrid V1 (GPT-4o) | 0.4943 | 0.6862 | 0.6475 | 0.6284 | 0.7558 | 31 |
| Presidio | 0.4933 | 0.5493 | **0.6680** | &mdash; | &mdash; | 86 |
| Piiranha | 0.4731 | 0.6501 | 0.6195 | &mdash; | &mdash; | 31 |
| NerGuard Base (no LLM) | 0.4175 | 0.6105 | 0.6076 | &mdash; | &mdash; | 33 |
| spaCy (en_core_web_trf) | 0.3607 | 0.4175 | 0.5527 | &mdash; | &mdash; | 144 |
| dslim/bert-base-NER | 0.3331 | 0.4821 | 0.6225 | &mdash; | &mdash; | 38 |

NerGuard Hybrid V2 ranks **1st on both F1-macro and F1-micro**, while achieving **2.1x lower latency** than Presidio (the closest competitor on F1-macro) and **3.5x lower latency** than spaCy. Compared to the base model alone, the hybrid pipeline provides a **+8.94 pt F1-macro gain** through selective LLM routing and regex validation, demonstrating that the entropy-gated architecture effectively targets the base model's weaknesses without degrading its strengths.

### V2 Ablation (over V1)

| Metric | V1 | V2 | Delta |
|---|---|---|---|
| F1-macro | 0.4943 | 0.5069 | +1.26 pts |
| F1-micro | 0.6862 | 0.7015 | +1.53 pts |
| Entity-F1 | 0.6475 | 0.6634 | +1.59 pts |
| Precision | 0.6284 | 0.6484 | +2.00 pts |
| Recall | 0.7558 | 0.7641 | +0.83 pts |

The V2 gains are primarily attributed to: (1) regex force-override for credit card numbers (CC F1: 0.46 &rarr; 0.95), (2) regex demotion for SSN validation (SSN F1: 0.31 &rarr; 0.62), and (3) a label assembly fix that enabled the full regex correction chain to take effect.

### GDPR Relevance

NerGuard is designed with GDPR data protection requirements in mind:

- **High recall by design**; The system is tuned to favor PII detection over non-detection (recall 0.7641). Under GDPR Art. 17 (right to erasure) and Art. 4 (definition of personal data), a missed PII entity constitutes a compliance risk. NerGuard's LLM prompt explicitly encodes this bias: *"When uncertain, prefer classifying as PII over O."*
- **20-class fine-grained taxonomy**; Covers GDPR-relevant categories including government IDs (SSN, passport, driver's license, tax ID), financial data (credit card, IBAN), contact information (email, phone), and demographic attributes (age, gender), enabling category-specific data handling policies.
- **Multilingual support**; Covers 8 European languages (EN, DE, FR, IT, ES, PT, NL, PL) through mDeBERTa's cross-lingual transfer, relevant for organizations operating across EU member states.
- **Auditable pipeline**; Each prediction carries provenance metadata: base model confidence, entropy score, routing decision, LLM response, and regex validation outcome, enabling full traceability for Data Protection Impact Assessments (DPIA).

### LLM Backend

NerGuard is designed for **fully local, on-premise deployment** via [Ollama](https://ollama.com/). The LLM routing component communicates through a standard OpenAI-compatible API, making it backend-agnostic. The benchmark results above use GPT-4o solely as a convenience for reproducible evaluation; in production the system runs entirely on local hardware with no external API dependency; a critical property for GDPR-compliant data processing where PII must not leave the organization's infrastructure.

The following local models have been tested and are recommended:

| Model | Parameters | VRAM | Backend |
|---|---|---|---|
| `gpt-oss:20b` | 20B | ~13 GB | Ollama |
| `llama3.1:8b` | 8B | ~5 GB | Ollama |
| `qwen2.5:7b` | 7B | ~5 GB | Ollama |

To use a local backend, start Ollama and pass `--llm-source ollama` to the benchmark runner or set the LLM source in the inference API:

```bash
# Local inference (no data leaves the machine)
uv run python -m src.benchmark.runner \
  --systems nerguard-hybrid-v2 --datasets nvidia-pii --samples 100 \
  --llm-source ollama --llm-model llama3.1:8b \
  --semantic-alignment alignments/default.json
```

## Getting Started

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
    print(f"{e['label']}: {e['text']} (conf: {e['confidence']:.2%})")
```

### Reproducing the Benchmark

```bash
uv run python -m src.benchmark.runner \
  --systems nerguard-hybrid-v2,nerguard-hybrid,nerguard-base,presidio,spacy,piiranha,bert-ner \
  --datasets nvidia-pii --samples 1000 --llm-model gpt-4o --batch-llm \
  --semantic-alignment alignments/default.json
```

## Repository Structure

```
src/
  core/            Constants, metrics, model loading, label mapping
  inference/       LLM router, entity router, regex validator, span assembler, prompts
  training/        Model training and validation
  benchmark/       Cross-system benchmark framework (runner, metrics, datasets, systems)
  optimization/    Threshold optimizer, ONNX quantization
  scripts/         CLI entry points (train, evaluate, demo, inference)
  utils/           I/O, logging, sample data
docs/              Technical notes, architecture diagrams, bibliography
experiments/       Benchmark results
alignments/        Semantic label alignment for cross-system evaluation
scripts/           Shell wrappers for reproducibility
```

## Citation

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
