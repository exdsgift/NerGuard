<div align="center">
  <h1>NerGuard</h1>
  <p><strong>Hybrid PII Detection with Entropy-Based LLM Routing</strong></p>
</div>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/W&B-FFBE00?style=flat&logo=weightsandbiases&logoColor=black" alt="W&B"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/exdsgift/NerGuard-0.3B">Model</a> &middot;
  <a href="https://huggingface.co/exdsgift/NerGuard-0.3B-onnx-int8">Quantized Model</a>
</p>

---

NerGuard is a PII (Personally Identifiable Information) detection system that combines a fine-tuned mDeBERTa-v3-base model with selective LLM routing. When the model's prediction entropy exceeds learned thresholds, uncertain tokens are routed to GPT-4o for disambiguation. This approach routes only **0.57%** of tokens while achieving **88.89%** correction accuracy, reducing API costs by **78%** compared to full routing.

The system detects 21 PII entity types across 8 European languages using BIO tagging.

## Results

### Validation Set

| Model | F1 | Latency (ms) |
|---|---|---|
| **NerGuard (Base)** | **0.904** | 22.4 |
| NerGuard (Hybrid) | 0.899 | 321.6 |
| GLiNER | 0.446 | 20.9 |
| Presidio | 0.315 | 10.1 |
| SpaCy | 0.142 | 8.8 |

### LLM Routing

| Metric | Value |
|---|---|
| Tokens Routed | 0.57% |
| Correction Accuracy | 88.89% |
| Cost Savings vs Full Routing | 78% |
| Help:Harm Ratio | 2.2:1 |

### Entity-Specific Improvements (with LLM)

| Entity | Without LLM | With LLM | Delta |
|---|---|---|---|
| Credit Card Number | 4.7% | 25.6% | +20.9% |
| Phone Number | 38.5% | 49.4% | +10.9% |
| Surname | 60.9% | 63.7% | +2.7% |
| Date | 82.3% | 84.3% | +2.0% |

### NVIDIA/Nemotron-PII Dataset

| Metric | Score |
|---|---|
| Overall Accuracy | 93.22% |
| Weighted F1 | 95.17% |
| Macro F1 | 35.06% |

*Evaluated on 1,000 samples.*

### Cross-Lingual Transfer

| Language | F1 Macro | Language | F1 Macro |
|---|---|---|---|
| Polish | 0.744 | English | 0.600 |
| Dutch | 0.615 | French | 0.561 |
| Italian | 0.604 | Portuguese | 0.538 |
| German | 0.600 | Spanish | 0.515 |

## Installation

```bash
git clone https://github.com/exdsgift/NerGuard.git
cd NerGuard
uv sync
```

## Usage

```python
from src.inference.tester import PIITester

tester = PIITester(model_path="exdsgift/NerGuard-0.3B")
entities = tester.get_entities("John Smith lives at 123 Main St. Email: john@email.com")

for e in entities:
    print(f"{e['label']}: {e['text']} (conf: {e['confidence']:.2%})")
```

## Commands

| Script | Description |
|---|---|
| `./scripts/setup.sh` | Install dependencies and download models |
| `./scripts/train.sh` | Train NerGuard model |
| `./scripts/evaluate.sh` | Run hybrid evaluation (baseline vs LLM routing) |
| `./scripts/demo.sh` | Interactive demo |
| `./scripts/inference.sh` | Run PII detection on text input |
| `./scripts/benchmark.sh` | Benchmark against GLiNER, Presidio, SpaCy |
| `./scripts/evaluate_nvidia.sh` | Evaluate on NVIDIA/Nemotron-PII |
| `./scripts/evaluate_multilingual.sh` | Cross-lingual evaluation (8 languages) |
| `./scripts/ablation_study.sh` | LLM routing ablation study |

## Models

| Model | Parameters | Format | Link |
|---|---|---|---|
| NerGuard-0.3B | 279M | PyTorch | [exdsgift/NerGuard-0.3B](https://huggingface.co/exdsgift/NerGuard-0.3B) |
| NerGuard-0.3B-onnx-int8 | 279M | ONNX INT8 | [exdsgift/NerGuard-0.3B-onnx-int8](https://huggingface.co/exdsgift/NerGuard-0.3B-onnx-int8) |

## Project Structure

```
src/
  core/            Constants, metrics, model loading, label mapping
  inference/       PIITester, LLM router, entity router, prompts
  training/        Model training and validation
  evaluation/      Benchmark, ablation study, multilingual, NVIDIA evaluation
  optimization/    Threshold optimization, ONNX quantization
  experiments/     LLM router experiment
  scripts/         CLI entry points (train, evaluate, demo, inference)
  utils/           I/O, logging, sample data
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
