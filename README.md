<div align="center">
  <h1>NerGuard</h1>
  <p>
    <strong>Hybrid PII Detection with Entropy-Based LLM Routing</strong><br>
    <em>Master's Thesis in Data Science</em><br>
    University of Verona, Department of Computer Science
  </p>
</div>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat&logo=weightsandbiases&logoColor=black" alt="W&B"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI"></a>
</p>

**NerGuard** is a PII (Personally Identifiable Information) detection system combining transformer-based NER with selective LLM routing. The system uses entropy-based uncertainty quantification to identify when local predictions are unreliable, routing only those cases to an LLM for disambiguation.

- Download the main model from HF: [exdsgift/NerGuard-0.3B](https://huggingface.co/exdsgift/NerGuard-0.3B)
- Download the quantized model from HF: [exdsgift/NerGuard-0.3B-onnx-int8](https://huggingface.co/exdsgift/NerGuard-0.3B-onnx-int8)

### Key Contributions

- **Hybrid Architecture**: Combines fast local inference with selective LLM routing
- **Entropy-Based Routing**: Uses Shannon entropy to detect model uncertainty
- **Production-Ready**: Sliding window processing, caching, and comprehensive validation
- **Benchmarking Suite**: Comparison against Presidio, GLiNER, and SpaCy


## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/NerGuard.git
cd NerGuard
pip install -e .
```

### Environment Setup

```bash
cp .env.example .env
# Add your API key for LLM routing (optional)
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### Run Demo

```bash
# Basic demo
python -m src.scripts.demo

# With custom text
python -m src.scripts.demo --text "Contact John Smith at john@email.com"

# With LLM routing enabled
python -m src.scripts.demo --llm-routing

# Detailed token analysis
python -m src.scripts.demo --detailed
```

### Python API

```python
from src.inference.tester import PIITester

# Initialize
tester = PIITester(llm_routing=False)

# Detect entities
text = "John Smith lives at 123 Main St. Email: john@email.com"
entities = tester.get_entities(text)

for e in entities:
    print(f"{e['label']}: {e['text']} (conf: {e['confidence']:.2%})")

# Redact text
redacted = tester.redact_text(text)
print(redacted)
```

---

## Architecture

```
    Input Text
         │
         ▼
┌─────────────────┐
│   Tokenizer     │  mDeBERTa-v3 tokenizer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sliding Window  │  Handles documents > 512 tokens
│ (stride: 382)   │  Overlap: 128 tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ mDeBERTa-v3     │  Token classification
│ Encoder         │  21 entity types (BIO scheme)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Uncertainty     │  Entropy + Confidence
│ Estimation      │  computation
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Certain   Uncertain
    │         │
    │    ┌────┴────┐
    │    │  LLM    │  Selective routing
    │    │ Router  │  with caching
    │    └────┬────┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ BIO Validation  │  Consistency check
└────────┬────────┘
         │
         ▼
   Output Labels
```

### Routing Criteria

LLM disambiguation is triggered when:
- `entropy > 0.583` AND `confidence < 0.787`

Thresholds optimized via grid search with bootstrap confidence intervals.

---

## Project Structure

```
NerGuard/
├── src/
│   ├── core/                    # Core utilities
│   │   ├── constants.py         # Configuration values
│   │   ├── model_loader.py      # Model loading utilities
│   │   ├── label_mapper.py      # Label mapping
│   │   └── metrics.py           # Entropy/confidence computation
│   │
│   ├── inference/               # Inference pipeline
│   │   ├── tester.py            # PIITester class
│   │   ├── llm_router.py        # LLM routing with cache
│   │   └── prompts.py           # LLM prompt templates
│   │
│   ├── training/                # Model training
│   │   ├── trainer.py           # Training loop with W&B
│   │   ├── encoder.py           # Model architecture
│   │   └── validator.py         # Validation utilities
│   │
│   ├── evaluation/              # Evaluation scripts
│   │   ├── benchmark.py         # Multi-model benchmark
│   │   ├── nvidia_evaluator.py  # NVIDIA dataset evaluation
│   │   └── hybrid_evaluator.py  # Hybrid system evaluation
│   │
│   ├── optimization/            # Optimization tools
│   │   ├── threshold_optimizer.py  # Grid search
│   │   └── quantizer.py         # ONNX quantization
│   │
│   ├── visualization/           # Plotting utilities
│   │   ├── style.py             # Publication style
│   │   ├── plots.py             # Core plots
│   │   ├── benchmark_plots.py   # Benchmark visualizations
│   │   └── optimization_plots.py
│   │
│   ├── scripts/                 # Runnable scripts
│   │   └── demo.py              # Interactive demo
│   │
│   └── utils/                   # Utilities
│       ├── colors.py            # Terminal colors
│       ├── io.py                # File I/O
│       ├── logging_config.py    # Logging setup
│       └── samples.py           # Test samples
│
├── models/                      # Trained models
├── data/                        # Datasets
├── plots/                       # Generated figures
└── notebooks/                   # Analysis notebooks
```

---

## Supported Entities

NerGuard detects **21 PII entity types** using BIO tagging:

| Category | Entities |
|----------|----------|
| **Person** | `GIVENNAME`, `SURNAME`, `TITLE` |
| **Location** | `CITY`, `STREET`, `BUILDINGNUM`, `ZIPCODE` |
| **Government ID** | `IDCARDNUM`, `PASSPORTNUM`, `DRIVERLICENSENUM`, `SOCIALNUM`, `TAXNUM` |
| **Financial** | `CREDITCARDNUMBER` |
| **Contact** | `EMAIL`, `TELEPHONENUM` |
| **Temporal** | `DATE`, `TIME` |
| **Demographic** | `AGE`, `SEX`, `GENDER` |

---

## Evaluation Results

### Performance on NVIDIA PII Dataset

| Metric | Score |
|--------|-------|
| Overall Accuracy | 93.22% |
| Weighted F1 | 95.17% |
| Macro F1 | 35.06% |

*Evaluated on 1,000 samples from NVIDIA/Nemotron-PII dataset with label alignment.*

### Hybrid System Analysis

The hybrid system combines base model inference with selective LLM routing:

- **LLM routing criteria**: entropy > 0.583 AND confidence < 0.787
- **Routing rate**: ~31% of tokens in uncertain predictions
- **Context window**: 400 characters for disambiguation

The LLM uses chain-of-thought reasoning to verify uncertain predictions:
1. Analyzes token context
2. Checks BIO consistency with previous label
3. Classifies PII type with reasoning

### Benchmark Comparison (3,000 samples)

| Model | F1-Score | Latency (ms) |
|-------|----------|--------------|
| **NerGuard (Base)** | **0.904** | 22.4 |
| NerGuard (Hybrid) | 0.899 | 321.6 |
| GLiNER | 0.446 | 20.9 |
| Presidio | 0.315 | 10.1 |
| SpaCy | 0.142 | 8.8 |

*NerGuard significantly outperforms open-source alternatives on PII detection.*

---

## Configuration

### Constants (`src/core/constants.py`)

```python
DEFAULT_MODEL_PATH = "./models/mdeberta-pii-safe/final"
DEFAULT_ENTROPY_THRESHOLD = 0.583
DEFAULT_CONFIDENCE_THRESHOLD = 0.787
MAX_CONTEXT_LENGTH = 512
OVERLAP = 128
```

### LLM Router

```python
from src.inference.llm_router import LLMRouter

router = LLMRouter(
    source="openai",        # or "ollama"
    model="gpt-4o-mini",
    enable_cache=True,
    cache_size=1000
)
```

---

## Training

```bash
# Train with default configuration
python -m src.training.trainer

# Custom configuration
python -m src.training.trainer \
    --model microsoft/mdeberta-v3-base \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5
```

Training tracked with Weights & Biases.

---

## Benchmarking

```bash
# Run full benchmark
python -m src.evaluation.benchmark

# NVIDIA dataset evaluation
python -m src.evaluation.nvidia_evaluator

# Hybrid system evaluation
python -m src.evaluation.hybrid_evaluator
```

---

## Citation

```bibtex
@mastersthesis{nerguard2026,
  title     = {NerGuard: Hybrid PII Detection with Entropy-Based LLM Routing},
  author    = {[Gabriele Durante]},
  year      = {2026},
  school    = {University of Verona},
  type      = {Master's Thesis},
  department = {Department of Computer Science}
}
```

---

## License

Academic research project. Contact the author for usage permissions.

---

<p align="center">
  <sub>Built with</sub>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/W&B-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" alt="W&B"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"></a>
</p>
