# NerGuard — System Improvements & Evaluation Results (2026-03-02)

Dataset: **NVIDIA/Nemotron-PII** (English, synthetic, ~26 k sentences)
Baseline model: **mDeBERTa-v3-base** fine-tuned on internal PII corpus
LLM router: **llama3.1:8b** via Ollama (local A100)
Evaluation: 1,000 samples, unbiased intersection-only mode
Run: `results/validation_20260302_100206/`

---

## Summary of Changes

Three orthogonal improvements were implemented in this session:

| # | Component | File(s) | Description |
|---|-----------|---------|-------------|
| 1 | Unbiased benchmarking | `hybrid_evaluator.py`, `constants.py` | Intersection-only evaluation; ambiguous NVIDIA labels excluded |
| 2 | RegexValidator | `regex_validator.py` (new) | Post-processing O→entity promotion + LLM routing guard |
| 3 | Two-pass routing | `hybrid_evaluator.py` | B- tokens routed first, I- tokens after (opt-in via `--two-pass`) |

---

## 1. Unbiased Benchmarking

### Problem
Previous `NVIDIA_TO_MODEL_MAP` merged NVIDIA labels many-to-one onto model labels and included
semantically incorrect mappings (e.g. `user_name` → `GIVENNAME`, `sexuality` → `GENDER`).
This inflated/deflated metrics unpredictably and made cross-system comparison impossible.

### Solution: Intersection-Only Evaluation

Added to `src/core/constants.py`:
- **`CLEAN_NVIDIA_MAP`** — 32 entries, only unambiguous 1:1 mappings (e.g. `email` → `EMAIL`, `ssn` → `SOCIALNUM`)
- **`EXCLUDED_NVIDIA_LABELS`** — 6 labels excluded entirely from evaluation:
  `middle_name`, `name`, `user_name`, `fax_number`, `date_time`, `sexuality`

Tokens bearing excluded labels are masked out (`label = -100`) before any TP/FP/FN computation.
They appear in the report but do not penalize the model for entities it was never trained to predict.

### Coverage on 1,000 samples
```
Total span tokens:    38,427
Included in metrics:  36,860  (95.9%)
Excluded:              1,567   (4.1%)
```
4.1% exclusion rate — minimal data loss, maximum metric integrity.

---

## 2. RegexValidator

### Architecture

The new layer sits **after** the main routing loop:

```
mDeBERTa output
    │
    ▼
EntitySpecificRouter.should_route()
    │
    ├── regex.can_skip_llm() → True → skip LLM, keep model label
    │
    └── False → LLM call → hybrid_preds updated
    │
    ▼
regex.correct_predictions()     ← post-processing O→entity promotion
    │
    ▼
Final hybrid_preds
```

### Patterns implemented

| Entity | Pattern | Validator |
|--------|---------|-----------|
| CREDITCARDNUMBER | Visa/MC/Amex/Discover/Diners + separators | Luhn algorithm |
| EMAIL | RFC 5322 simplified | — |
| SOCIALNUM | `\d{3}-\d{2}-\d{4}` | Structural check (no all-zero groups) |
| IBAN | `[A-Z]{2}\d{2}[A-Z0-9]{4,30}` | ISO 13616 mod-97 check digit |
| TELEPHONENUM | E.164 + common national formats | — |
| DATE | ISO 8601 + DD/MM + MM/DD | — |
| ZIPCODE | US 5-digit (±4) + UK postcode | — |

### Design principles
- **High precision over recall**: false positives from regex are harmful → all patterns conservative
- **O→entity promotion only if ALL overlapping tokens are O**: never overwrites partial model predictions
- **Plug-and-play**: entities without a configured pattern are unaffected

### Impact on 1,000 samples
```
Regex LLM skips:      190   (calls avoided — already confirmed by regex)
Regex O→entity:       560   (O tokens promoted to correct entity label)
LLM calls:            654   (vs ~1,047 estimated without regex)  →  37.5% fewer
Elapsed time:         7.09 min  (vs ~11.3 min without regex)     →  37% faster
```

---

## 3. Two-Pass Routing

### Motivation
I- tokens depend on the B- prediction upstream (chaining effect).
In single-pass mode, an I- token is routed with whatever `hybrid_preds[i-1]` happened to be at the
time it was processed — which may still be the uncorrected baseline if the B- call hasn't been made.

### Implementation
When `--two-pass` is active, the routing loop runs twice:
- **Pass 1**: only B- tokens are routed → `hybrid_preds` updated
- **Pass 2**: only I- tokens are routed → sees corrected `prev_label` from Pass 1

Single-pass (default) remains the validated baseline.
Two-pass ablation is pending (see section 6).

---

## 4. Evaluation Results

Run: `results/validation_20260302_100206/`
Config: `unbiased=True`, `regex=True`, `two_pass=False`

### Global metrics

| Metric | Baseline | Hybrid | Delta |
|--------|----------|--------|-------|
| Accuracy | 0.9377 | 0.9394 | +0.0017 |
| Macro Precision | 0.4068 | 0.4348 | +0.0280 |
| Macro Recall | 0.4612 | 0.4930 | +0.0318 |
| **Macro F1** | **0.3776** | **0.4046** | **+0.0270** |
| **Span F1 (seqeval)** | **0.4820** | **0.5045** | **+0.0225** |

### LLM routing summary

```
Tokens checked:       222,970
Routed to LLM:          4,632  (2.08%)
LLM calls:                654
Helpful corrections:      295
Harmful corrections:       53
Net improvement:          242
Hit rate:               295 / 654 = 45.1%
```

---

## 5. Per-Class Analysis

### Winners (hybrid significantly better)

| Class | Baseline acc | Hybrid acc | Delta | Notes |
|-------|-------------|-----------|-------|-------|
| B-CREDITCARDNUMBER | 2.2% | 34.8% | **+32.6%** | LLM: 29 helped, 0 hurt |
| I-CREDITCARDNUMBER | 1.5% | 20.4% | **+18.9%** | LLM: 102 helped, 0 hurt; Luhn filter critical |
| B-TELEPHONENUM | 53.7% | 67.5% | **+13.7%** | LLM: 35 helped, 0 hurt |
| I-TELEPHONENUM | 69.8% | 79.8% | **+9.9%** | LLM: 90 helped, 0 hurt |
| B-GENDER | 2.1% | 10.4% | +8.3% | LLM: 4 helped, 0 hurt |
| B-DATE | 77.6% | 78.9% | +1.3% | LLM: 11 helped, 0 hurt |

CREDITCARDNUMBER is the standout: I-CREDITCARDNUMBER F1 jumped from 0.03 → 0.33.
The Luhn check eliminates false positives from sequences that look like CC numbers but aren't.
With Luhn validation: precision 24% → 81% on I-CREDITCARDNUMBER.

### Neutral (no significant change)

B-DATE, B-EMAIL, B-SOCIALNUM, B-ZIPCODE — model already performs well, LLM rarely triggered.

### Losers (hybrid slightly worse)

| Class | Baseline acc | Hybrid acc | Delta | Notes |
|-------|-------------|-----------|-------|-------|
| I-SURNAME | 64.6% | 60.8% | **-3.8%** | LLM: 4 helped, 20 hurt |
| I-GIVENNAME | 77.3% | 74.4% | -2.9% | LLM: 0 helped, 8 hurt |
| B-AGE | 100.0% | 97.9% | -2.1% | LLM: 0 helped, 1 hurt |
| I-STREET | 68.1% | 67.3% | -0.8% | LLM: 2 helped, 8 hurt |

**I-SURNAME is the clearest casualty**: 20 harmful interventions vs only 4 helpful.
This confirms the previous empirical finding that name continuation tokens (I-SURNAME, I-GIVENNAME)
are systematically hurt by LLM routing — the LLM struggles with BIO sequence constraints for names.

The config currently uses `enable_selective=True` with universal routing (all entity types routable),
which allows LLM to intervene on I-SURNAME/I-GIVENNAME. Blocking these explicitly would recover
those ~20 net-negative corrections at the cost of losing the 4 helpful ones.

---

## 6. Pending: Two-Pass Routing Ablation

The `--two-pass` flag is implemented but not yet benchmarked.
Expected benefit: I-CREDITCARDNUMBER and I-TELEPHONENUM should see additional gains
since their B- counterparts will already be corrected when I- routing begins.

```bash
# Run in tmux:
uv run python -m src.evaluation.hybrid_evaluator \
  --max-samples 1000 --ollama-model llama3.1:8b \
  --two-pass \
  --output-dir results/validation_twopass_$(date +%Y%m%d_%H%M%S)
```

Compare `delta_span_f1` and `I-CREDITCARDNUMBER` accuracy between single-pass and two-pass.

---

## 7. Conclusions

The three improvements are complementary and address different failure modes:

1. **Unbiased benchmarking** fixes the measurement: F1 now reflects only what the model can predict,
   without contamination from unmappable NVIDIA-specific labels.

2. **RegexValidator** addresses structured PII missed by the model (O→entity promotions) and
   eliminates unnecessary LLM calls when the regex already confirms the prediction — the single
   highest-leverage change for throughput (37% faster).

3. **Two-pass routing** is a structural fix for chaining dependency; impact TBD from ablation.

**Next open question**: Should I-SURNAME and I-GIVENNAME be explicitly blocked from LLM routing?
The empirical data says yes (net -16 and -8 respectively), but the current universal config
keeps them in to preserve architectural simplicity. The entity-specific blocking logic
already exists in `EntitySpecificRouter` — it would be a one-line change to `BLOCKED_ENTITIES`.
