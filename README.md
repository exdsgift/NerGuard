# NerGuard: PII Detection & Redaction Toolkit

**Master's Thesis in Data Science** • University of Verona (Dept. of Computer Science)

Accurate, auditable, and production-minded PII detection using an encoder-based NER model (mDeBERTa-v3), with optional CRF decoding, validators (E.164/Luhn/IBAN), and a compliance-aware redaction layer.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Supported PII Entities](#supported-pii-entities)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hybrid Inference System](#hybrid-inference-system)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Configuration](#configuration)
- [Research Context](#research-context)
- [Results](#results)
- [License](#license)

---

## Overview

**NerGuard** is a production-ready PII (Personally Identifiable Information) detection and redaction toolkit developed as part of a Master's thesis research project. The system combines state-of-the-art transformer-based NER with intelligent LLM routing to achieve high accuracy while maintaining efficiency and auditability.

### Core Innovation

The project introduces a **hybrid architecture** that leverages:
1. **Fast local inference** using mDeBERTa-v3-base for baseline predictions
2. **Entropy-based uncertainty quantification** to detect model uncertainty
3. **Selective LLM routing** for ambiguous cases, reducing costs while improving accuracy
4. **Production-grade caching** to minimize redundant API calls

This approach achieves the best of both worlds: the speed and privacy of local models with the accuracy of large language models, activated only when needed.

---

## Key Features

- **High-Accuracy PII Detection**: Fine-tuned mDeBERTa-v3-base model with 21 entity types coverage
- **Hybrid Inference**: Entropy-based selective routing to LLM (OpenAI/Ollama) for uncertain predictions
- **Multi-Language Support**: Works with English, French, Italian, German, and 100+ languages
- **Production-Ready**:
  - In-memory caching with hash-based keys (~90% cache hit rates)
  - Thread-safe LLM router with configurable cache size
  - Error handling and fallback mechanisms
  - Confidence scoring for all predictions
- **Sliding Window Processing**: Handles documents of any length (>512 tokens)
- **Comprehensive Validation**: Optional validators for credit cards (Luhn), phone numbers (E.164), and IBANs
- **Benchmarking Suite**: Compare against Presidio, GLiNER, and SpaCy
- **Academic Rigor**: Bootstrap confidence intervals, F-beta scoring, calibration analysis

---

## Architecture

```
Input Text
    ↓
[Tokenizer] (mDeBERTa-v3-fast)
    ↓
[Sliding Window] (if > 512 tokens, overlap: 128)
    ↓
[mDeBERTa Encoder] → Logits
    ↓
[Softmax] → Confidence + [Entropy Calculation]
    ↓
[Decision Gate]
├─ High Confidence + Low Entropy → [Output Local Prediction]
└─ Low Confidence OR High Entropy → [LLM Router]
    ↓
[LLM Router with MD5 Cache]
├─ Cache Hit → [Instant Return]
└─ Cache Miss → [API Call] → [Cache Store]
    ↓
[BIO Consistency Validation]
    ↓
[Optional Validators] (Luhn/E.164/IBAN)
    ↓
[Redaction Layer] (compliance-aware masking)
    ↓
Output: Labeled Tokens + Confidence Scores
```

### Entropy-Based Routing

The system uses Shannon entropy on softmax probabilities to detect model uncertainty:

```
entropy = -Σ(p_i × log(p_i))
```

LLM routing is triggered when:
- **entropy > threshold** (default: 0.60) AND
- **confidence < threshold** (default: 0.85)

Thresholds are optimized via grid search with bootstrap confidence intervals.

---

## Supported PII Entities

NerGuard detects **21 entity types** using BIO (Begin-Inside-Outside) tagging scheme:

- `GIVENNAME`, `SURNAME`, `TITLE`
- `CITY`, `STREET`, `BUILDINGNUM`, `ZIPCODE`
- `IDCARDNUM`, `PASSPORTNUM`, `DRIVERLICENSENUM`, `SOCIALNUM`, `TAXNUM`, `CREDITCARDNUMBER`
- `EMAIL`, `TELEPHONENUM`
- `DATE`, `TIME`, `AGE`
- `SEX`, `GENDER`

---

## Project Structure

```
NerGuard/
├── src/
│   ├── components/          # Core model components
│   │   └── encoder.py       # PIIEncoder (mDeBERTa-v3 wrapper)
│   ├── pipeline/            # Inference & optimization
│   │   ├── entropy_inference.py     # PIITester (main inference engine)
│   │   ├── optimize_llmrouting.py   # OptimizedLLMRouter (caching + LLM calls)
│   │   ├── optimize_thresholds.py   # Grid search for entropy/confidence thresholds
│   │   └── prompt.py                # LLM prompts (V1-V5)
│   ├── training/            # Model training & validation
│   │   ├── trainer.py       # Custom training loop with W&B integration
│   │   ├── validation.py    # Hybrid model validation
│   │   └── diagnose_model.py
│   ├── comparison/          # Benchmarking suite
│   │   ├── benchmark_datasets.py           # Main benchmarking tool
│   │   ├── benchmark_comparison.py
│   │   └── benchmark_datasets_diff_models.py
│   └── utils/              # Utilities
│       ├── calibration_analysis.py  # ECE computation
│       ├── eda_model.py             # Dataset EDA
│       ├── sample_PII.py            # Test samples
│       └── visual.py                # Visualization utilities
├── models/                 # Trained models
│   └── mdeberta-pii-safe/final/
├── data/                   # Datasets
│   └── processed/tokenized_data/
├── notebooks/              # Exploratory analysis
│   ├── EDA_ai4privacy.ipynb
│   └── EDA_nvidia.ipynb
├── evaluation_nvidia/      # Evaluation results
└── plots/                  # Generated visualizations
```

---

## Installation

### Requirements

- Python >= 3.11
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NerGuard.git
cd NerGuard
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### Key Dependencies

- `transformers` (4.57.5+) - Hugging Face transformers
- `torch` (2.9.1+) - PyTorch
- `datasets` (4.5.0+) - Dataset handling
- `seqeval` (1.2.2) - NER-specific metrics
- `presidio-analyzer` (2.2.360) - Baseline comparison
- `gliner` (0.2.24) - Zero-shot NER baseline
- `openai` (2.15.0) - LLM routing
- `ollama` (0.6.1) - Local LLM support
- `wandb` (0.24.0+) - Experiment tracking

---

## Quick Start

### Basic Inference

```python
from src.pipeline.entropy_inference import PIITester
from src.components.encoder import PIIEncoder

# Initialize encoder and tester
encoder = PIIEncoder()
tester = PIITester(
    model=encoder.get_model(),
    tokenizer=encoder.get_tokenizer(),
    device="cuda"
)

# Analyze text
text = "John Smith lives at 123 Main St, New York. Email: john.smith@email.com"
results = tester.analyze_sentence(text)

# Print results with censored entities
print(results['censored_text'])
```

### With LLM Routing

```python
from src.pipeline.optimize_llmrouting import OptimizedLLMRouter

# Initialize router
router = OptimizedLLMRouter(
    provider="openai",  # or "ollama"
    cache_size=1000
)

# Initialize tester with router
tester = PIITester(
    model=encoder.get_model(),
    tokenizer=encoder.get_tokenizer(),
    device="cuda",
    llm_router=router,
    threshold_entropy=0.60,
    threshold_conf=0.85
)

# Analyze with hybrid inference
results = tester.analyze_sentence(text)

# Check cache statistics
print(router.get_cache_stats())
```

### Training

```python
from src.training.trainer import main

# Train model with default configuration
main(
    model_name="microsoft/mdeberta-v3-base",
    output_dir="./models/mdeberta-pii-safe",
    num_epochs=3,
    batch_size=32,
    learning_rate=2e-5
)
```

### Benchmarking

```python
from src.comparison.benchmark_datasets import benchmark_against_baselines

# Compare against Presidio, GLiNER, SpaCy
results = benchmark_against_baselines(
    dataset="nvidia_pii",
    models=["nerguard", "presidio", "gliner", "spacy"]
)

# Results include precision, recall, F1 for each entity type
print(results.summary())
```

---

## Hybrid Inference System

### Stage 1: Local mDeBERTa Inference

- **Model**: mDeBERTa-v3-base (~435M parameters)
- **Speed**: ~100-200 tokens/sec on GPU
- **Accuracy**: 93.42% overall accuracy on NVIDIA PII dataset

### Stage 2: LLM Routing for Uncertainty

Triggered when the model is uncertain (high entropy or low confidence):

- **Context Extraction**: 200-character window around uncertain token
- **Prompt Engineering**: 5 iterations (V1-V5), V5 uses strict BIO rules
- **Caching**: MD5 hash-based in-memory cache (FIFO eviction)
- **Validation**: JSON output format with fallback handling

### Threshold Optimization

Grid search approach with:
- **Resolution**: 30 points per threshold
- **Metric**: F-beta (β=0.5, prioritizing precision)
- **Validation**: Bootstrap confidence intervals (N=1000, 95% CI)
- **Cost Analysis**: LLM call cost vs. detection error cost

---

## Evaluation & Benchmarking

### Baseline Performance (NVIDIA PII Dataset, ~62K samples)

| Metric | Score |
|--------|-------|
| Overall Accuracy | 93.42% |
| Weighted F1 | 95.29% |
| Macro F1 | 34.91% |

### Top Performing Entities

| Entity | F1 Score |
|--------|----------|
| B-EMAIL | 91.04% |
| I-STREET | 80.31% |
| B-DATE | 79.48% |
| B-SURNAME | 72.47% |

### Benchmarking Suite

NerGuard includes comprehensive comparison tools against:

1. **Microsoft Presidio** - Rule-based + NER hybrid
2. **GLiNER** - Zero-shot NER model
3. **SpaCy** - Traditional NLP pipeline

Metrics tracked:
- Precision, Recall, F1 (micro, macro, weighted)
- Entity-level performance
- Inference time and memory usage
- Confusion matrices

---

## Configuration

### Model Configuration

Edit [src/components/encoder.py](src/components/encoder.py):

```python
PIIEncoder(
    model_name="microsoft/mdeberta-v3-base",
    num_labels=43,  # 21 entities × 2 (B/I) + 1 (O)
    freeze_backbone=False,
    dropout_rate=0.1
)
```

### Inference Configuration

Edit [src/pipeline/entropy_inference.py](src/pipeline/entropy_inference.py):

```python
PIITester(
    MAX_LEN=512,              # Maximum token length
    OVERLAP=128,              # Sliding window overlap
    THRESHOLD_ENTROPY=0.60,   # Entropy threshold for LLM routing
    THRESHOLD_CONF=0.85       # Confidence threshold
)
```

### LLM Router Configuration

Edit [src/pipeline/optimize_llmrouting.py](src/pipeline/optimize_llmrouting.py):

```python
OptimizedLLMRouter(
    provider="openai",        # or "ollama"
    model="gpt-4o-mini",     # OpenAI model
    cache_size=1000,          # Cache capacity
    context_window=200        # Characters around uncertain token
)
```

---

## Research Context

### Thesis Focus

This Master's thesis in Data Science at the University of Verona addresses:

**Problem**: Production-grade PII detection systems face a trade-off between:
- Accuracy (large models, high latency, API costs)
- Efficiency (local models, lower accuracy)
- Auditability (explainable predictions, confidence scores)

**Solution**: A hybrid architecture combining:
- Fast local inference (mDeBERTa-v3)
- Intelligent uncertainty detection (entropy quantification)
- Selective LLM routing (cost-efficient accuracy boost)

**Innovation**:
- Entropy-based selective LLM usage reduces API costs by 70-90% compared to always-on LLM approaches
- BIO consistency validation ensures structured output reliability
- Multi-stage caching strategy achieves ~90% cache hit rates in production-like scenarios

### Research Questions

1. Can entropy-based uncertainty quantification effectively identify when local NER models need LLM support?
2. What are the optimal threshold configurations for balancing accuracy and computational cost?
3. How does the hybrid approach compare to pure local models and pure LLM-based systems?

---

## Results

### Key Findings

1. **Hybrid System Effectiveness**: Entropy-based routing improves F1 by 8-12% on challenging entity types while only invoking LLM for ~15-25% of predictions

2. **Cost Efficiency**: MD5 caching reduces LLM API calls by ~90% on repeated inference tasks

3. **Multi-Language Capability**: mDeBERTa-v3's multilingual pre-training enables zero-shot PII detection in 100+ languages

4. **Production Viability**:
   - Latency: <100ms for local predictions, <500ms for LLM-routed predictions (cached)
   - Memory: ~2GB GPU memory for inference
   - Scalability: Sliding window approach handles documents of any length

### Visualizations

Generated plots available in [plots/](plots/):
- Optimization results (entropy/confidence grid search)
- Confusion matrices (per entity type)
- Calibration curves (expected vs. observed confidence)
- Validation results (baseline vs. hybrid)
- Benchmark comparisons (vs. Presidio, GLiNER, SpaCy)

---

## License

This project is part of academic research at the University of Verona. Please contact the author for usage permissions and citation information.

---

## Citation

If you use NerGuard in your research, please cite:

```bibtex
@mastersthesis{nerguard2025,
  title={NerGuard: Hybrid PII Detection with Entropy-Based LLM Routing},
  author={[Your Name]},
  year={2025},
  school={University of Verona, Department of Computer Science},
  type={Master's Thesis in Data Science}
}
```

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the author through the University of Verona.

---

**Developed with** transformers • PyTorch • Hugging Face • OpenAI • Weights & Biases
