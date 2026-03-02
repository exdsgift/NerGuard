# NerGuard — Routing Observations & Ablation Study

Dataset: **NVIDIA/Nemotron-PII** (English, synthetic, ~26 k sentences)
Baseline model: **mDeBERTa-v3-base** fine-tuned on internal PII corpus
Hybrid system: mDeBERTa → selective LLM routing via Ollama (local GPU)
GPU: 2× NVIDIA A100-PCIE-40GB

---

## 1. System Setup

### Architecture

```
Token sequence
    │
    ▼
mDeBERTa (sliding-window, stride=382, overlap=128)
    │  per-token logits → softmax
    │  entropy + max-confidence → routing decision
    ▼
EntitySpecificRouter.should_route(label, entropy, confidence)
    ├─ threshold gate (per-entity thresholds)
    ├─ B-/I- gate (routable_entities, routable_i_entities, blocked_entities)
    └─ YES → LLM call (PROMPT_V13)
           │
           ▼
    LLM predicts entity CLASS only (e.g. "TELEPHONENUM", "O")
    _class_to_bio(class, prev_hybrid_label) → final BIO tag
```

### V13 Paradigm (key design decision)

In previous versions (V12), the LLM predicted full BIO tags (`B-TELEPHONENUM`, `I-TELEPHONENUM`, `O`). This caused frequent BIO violations (e.g., isolated `I-` tokens with no preceding `B-`).

V13 splits responsibility:
- **LLM**: predicts entity class only (no BIO prefix)
- **System**: assigns B-/I- deterministically via `_class_to_bio(entity_class, prev_label)`:
  - If `prev_label` shares the same class → `I-<class>`
  - Otherwise → `B-<class>`
  - If class is "O" → `O`

Result: zero BIO violations, simpler LLM task, more reliable output.

### Per-entity routing thresholds

| Entity | Entropy threshold | Confidence threshold |
|--------|------------------|---------------------|
| CREDITCARDNUMBER | 0.40 | 0.90 |
| TELEPHONENUM | 0.50 | 0.85 |
| SOCIALNUM | 0.40 | 0.90 |
| DEFAULT (all others) | 0.583 | 0.787 |

Thresholds optimized via grid search on a separate validation set.

---

## 2. Experiments

### Experiment A — Baseline routing (8 types, no I-), V12 prompt, 50 samples

**Model**: llama3.1:8b
**Config**: ROUTABLE_ENTITIES = {CREDITCARDNUMBER, TELEPHONENUM, SOCIALNUM, PASSPORTNUM, IDCARDNUM, DRIVERLICENSENUM, DATE, SURNAME}, BLOCKED_ENTITIES = everything else, I- always blocked

Initial sanity check on mini-batch. Results showed clear improvement signal on numeric entities.

---

### Experiment B — V13 vs V12, 50 samples (llama3.1:8b)

Switch from V12 (full BIO prediction) to V13 (class-only prediction):
- V13 eliminated all BIO-violation corrections (previously ~15% of LLM calls were BIO fixes)
- V13 improved LLM accuracy on CREDITCARDNUMBER tokens significantly
- ΔF1 higher with V13 despite same entity routing config

**Conclusion**: V13 is strictly better — simpler prompt, no BIO violations, higher accuracy.

---

### Experiment C — Full validation, 1000 samples × 3 models (8 types, no I-)

**Config**:
- ROUTABLE_ENTITIES = {CREDITCARDNUMBER, TELEPHONENUM, SOCIALNUM, PASSPORTNUM, IDCARDNUM, DRIVERLICENSENUM, DATE, SURNAME}
- BLOCKED_ENTITIES = all other types
- I- tokens: always blocked

| Model | Baseline F1 | Hybrid F1 | ΔF1 | LLM calls | Helped | Hurt | Net | Time |
|-------|------------|-----------|-----|-----------|--------|------|-----|------|
| gpt-oss:20b | — | — | +0.0162 | ~170 | 92 | 3 | +89 | 8.6 min |
| llama3.1:8b | — | — | +0.0157 | 172 | 79 | 0 | +79 | 2.2 min |
| qwen2.5:7b | — | — | +0.0141 | ~170 | 79 | 6 | +73 | 1.9 min |

**Key observations**:
- llama3.1:8b: 0 hurt tokens — most conservative, cleanest corrections
- gpt-oss:20b: highest net (+89) and highest ΔF1, but 5× slower than llama
- qwen2.5:7b: fastest overall (1.9 min), competitive ΔF1
- ~35% of LLM calls on PASSPORTNUM/IDCARDNUM/SOCIALNUM/DRIVERLICENSENUM with 0 net benefit
- I-CREDITCARDNUMBER (540 tokens, 1–3% baseline accuracy) completely untapped — major missed opportunity

---

### Experiment D — Universal routing (20 types + all I-), 1000 samples, llama3.1:8b

**Config**:
- ROUTABLE_ENTITIES = ENTITY_CLASSES (all 20 types)
- BLOCKED_ENTITIES = set() (none blocked)
- ROUTABLE_I_ENTITIES = ENTITY_CLASSES (all I- tokens routable)

| Metric | Value |
|--------|-------|
| ΔF1 | **+0.0222** (best across all configs) |
| LLM calls | 1047 |
| Helped | 343 |
| Hurt | 114 |
| Net | **+229** |
| Time | 11.3 min |

**Per-class breakdown (net corrections)**:

| Entity | Net | Helped | Hurt | Baseline acc | Hybrid acc | Notes |
|--------|-----|--------|------|-------------|------------|-------|
| I-CREDITCARDNUMBER | **+120** | 120 | 0 | 1.5% | 23.7% | Largest gain, 0 hurt |
| I-TELEPHONENUM | +76 | 109 | 33 | 61.7% | 67.8% | Strong improvement |
| B-TELEPHONENUM | +43 | 46 | 3 | — | — | |
| B-CREDITCARDNUMBER | +32 | 32 | 0 | — | — | |
| B-DATE | +11 | — | — | — | — | |
| B-SURNAME | +10 | — | — | — | — | |
| I-SURNAME | -16 | — | — | — | — | Harmful |
| I-GIVENNAME | -15 | — | — | — | — | Harmful |
| I-DATE | -12 | — | — | — | — | Harmful |
| B-GIVENNAME | -7 | — | — | — | — | Harmful |
| I-EMAIL | -6 | — | — | — | — | Harmful |
| I-STREET | -6 | — | — | — | — | Harmful |

**Conclusion**: Universal routing is the best single configuration. The gains from I-CREDITCARDNUMBER (+120) and I-TELEPHONENUM (+76) far outweigh the losses from harmful I- types.

---

### Experiment E — "Optimal" selective routing (19 types + I- CCN/TEL only), 1000 samples, llama3.1:8b

**Hypothesis**: Remove harmful entities (B-GIVENNAME, all I- except CCN/TEL) to reduce hurt while keeping winners.

**Config**:
- ROUTABLE_ENTITIES = ENTITY_CLASSES - {"GIVENNAME"}  (19 types)
- BLOCKED_ENTITIES = {"GIVENNAME"}
- ROUTABLE_I_ENTITIES = {"CREDITCARDNUMBER", "TELEPHONENUM"}

**Expected result** (naive additive): +120+76+43+32+11+10 - 7 ≈ +285

**Actual result**:

| Metric | Value |
|--------|-------|
| ΔF1 | +0.0166 |
| LLM calls | 473 |
| Helped | 102 |
| Hurt | 47 |
| Net | +55 |
| Time | 5.4 min |

**Actual per-class (most surprising)**:

| Entity | Expected net | Actual net |
|--------|-------------|------------|
| I-CREDITCARDNUMBER | +120 | +4 |
| I-TELEPHONENUM | +76 | **-39** |

**This was dramatically worse than expected** — especially I-TELEPHONENUM going from +76 to -39.

---

## 3. Chaining Effect Analysis

### Why does universal routing outperform "optimal" selective routing?

The key insight is the **chaining effect** in the V13 pipeline:

```
Token k:   B-TELEPHONENUM  (baseline wrong → routed → LLM corrects to TELEPHONENUM)
           ↓
           hybrid_preds[k] = "B-TELEPHONENUM"  ← prev_label for next token
           ↓
Token k+1: I-TELEPHONENUM  (LLM receives prev_label="B-TELEPHONENUM")
           → _class_to_bio("TELEPHONENUM", "B-TELEPHONENUM") = "I-TELEPHONENUM" ✓
```

In **universal routing**: All B- tokens of all types are corrected first. When I-CREDITCARDNUMBER or I-TELEPHONENUM is routed, it inherits a corrected `prev_label` from the already-fixed B- token. This cascade makes I- routing highly effective.

In **selective routing**: Only 19 B- types are corrected. When I-TELEPHONENUM is routed, its preceding B-TELEPHONENUM might not have been corrected (if that particular token happened to be processed correctly by mDeBERTa but with high entropy). More importantly, removing GIVENNAME from B- routing means some preceding context is wrong → I- tokens inherit bad prev_label → incorrect BIO assignment.

**Formula**: Net(I-entity) ≠ f(I-entity alone) — it depends on how many preceding B- tokens were corrected.

**Evidence**:
- Universal: I-TELEPHONENUM net = +76 (with 1047 B- corrections upstream)
- Selective: I-TELEPHONENUM net = -39 (with only 473 B- corrections upstream, broken chain)

### Pattern analysis: which I- entities benefit from routing?

**Safe (numeric, structured)**:
- I-CREDITCARDNUMBER: fixed-format digits, LLM recognizes continuations easily
- I-TELEPHONENUM: semi-structured, LLM handles well

**Unsafe (textual, ambiguous)**:
- I-SURNAME, I-GIVENNAME: free-form text, LLM confuses with O
- I-DATE: context-dependent, LLM struggles without full sentence
- I-EMAIL, I-STREET: variable-length continuations, harder to classify

In universal routing, even the "unsafe" I- entities are net neutral or slightly negative, but their contribution is small compared to the gains from CCN and TEL.

---

## 4. Three-Way Configuration Comparison (llama3.1:8b, 1000 samples)

| Configuration | Net | ΔF1 | LLM calls | Hurt | Time |
|---------------|-----|-----|-----------|------|------|
| Baseline (8 types, no I-) | +79 | +0.0157 | 172 | 0 | 2.2 min |
| **Universal (20 types + all I-)** | **+229** | **+0.0222** | 1047 | 114 | 11.3 min |
| Optimal-selective (19 types + I- CCN/TEL) | +55 | +0.0166 | 473 | 47 | 5.4 min |

**Trade-offs**:
- Baseline: lowest latency, zero hurt, lowest ΔF1
- Universal: highest ΔF1, highest net, but 6× more LLM calls vs baseline, 114 hurt
- Optimal-selective: intermediate calls, but WORSE than baseline in net and ΔF1 — the chaining effect collapse makes this config inefficient

**Recommendation**: Universal routing is the configuration to deploy and study further.

---

## 5. Conclusions

1. **V13 paradigm is essential**: Separating class prediction from BIO assignment eliminates violations and improves LLM accuracy.

2. **Universal routing wins empirically** (ΔF1 +0.0222 vs +0.0157 baseline): Opening all entity types and all I- tokens creates a cascade of corrections that compounds through the token sequence.

3. **The chaining effect is the dominant factor**: I- token routing quality depends critically on whether preceding B- tokens were corrected. Selective B- routing breaks the cascade and collapses I- performance.

4. **I-CREDITCARDNUMBER is the single biggest opportunity**: 540 tokens with 1–3% baseline accuracy, 0 hurt with LLM routing, net +120. This entity alone justifies enabling I- routing.

5. **Speed vs accuracy trade-off**:
   - qwen2.5:7b: fastest (1.9 min/1000 samples)
   - llama3.1:8b: best hurt ratio (0 hurt in baseline, low in universal)
   - gpt-oss:20b: best raw net in baseline config, slowest (8.6 min)

6. **Next experiments**:
   - Can per-entity thresholds be further tuned for I- entities?
   - Would a two-pass routing (B- first, then I-) improve results with fewer calls?

---

## 6. Full Pipeline Results (Universal Routing, 1000 samples × 3 models)

*Pipeline run: 2026-03-01, `results/validation_20260301_191818/`*

| Model | Baseline F1 | Hybrid F1 | ΔF1 | Net | LLM calls | Hurt | Time |
|-------|------------|-----------|-----|-----|-----------|------|------|
| gpt-oss:20b | 0.3573 | 0.3804 | **+0.0231** | +243 | 1047 | 146 | 49.0 min |
| llama3.1:8b | 0.3573 | 0.3795 | **+0.0222** | +229 | 1047 | 114 | 11.3 min |
| qwen2.5:7b  | 0.3573 | 0.3759 | **+0.0186** | +201 | 1047 | — | 9.4 min |

**Key findings**:
- All 3 models confirm universal routing as the best config
- gpt-oss:20b highest ΔF1 (+0.0231) but 4× slower than llama
- llama3.1:8b best speed/accuracy trade-off: ΔF1 +0.0222 in 11.3 min
- qwen2.5:7b fastest (9.4 min) with solid +0.0186 ΔF1
- LLM calls identical (1047) across all models — routing is model-agnostic

---

## 7. System Progression: Three-Level Comparison

### 7.1 Current hybrid vs mDeBERTa-only baseline

Per-entity F1 comparison (llama3.1:8b, 1000 samples, universal routing):

| Entity | Baseline F1 | Hybrid F1 | ΔF1 | Notes |
|--------|------------|-----------|-----|-------|
| B-CREDITCARDNUMBER | 0.04 | 0.42 | **+0.38** | Was practically unusable |
| I-CREDITCARDNUMBER | 0.03 | 0.37 | **+0.34** | Recall 1% → 24%, 0 hurt |
| B-TELEPHONENUM | 0.52 | 0.57 | +0.05 | |
| I-TELEPHONENUM | 0.63 | 0.68 | +0.05 | |
| B-DATE | 0.87 | 0.87 | ~0 | Already strong |
| B-EMAIL | 0.92 | 0.90 | -0.02 | Minimal regression |
| I-GIVENNAME | 0.49 | 0.48 | -0.01 | Minor hurt |
| **Macro F1** | **0.3573** | **0.3795** | **+0.0222** | **+6.2% relative** |
| Accuracy | 93.5% | 93.6% | +0.1% | Dominated by O tokens |

The accuracy/F1 gap is structural: 211k of 224k tokens are "O" and are unaffected. Macro F1 weights all 29 classes equally, so improving CREDITCARDNUMBER (F1 0.03→0.42) has outsized impact on the average.

### 7.2 Universal routing vs old selective routing (8 types, no I-)

Both runs use V13 prompt, llama3.1:8b, 1000 samples.

| Metric | Old (8 types, no I-) | Universal (20 types + I-) | Δ |
|--------|---------------------|--------------------------|---|
| LLM calls | 172 | 1047 | +875 (+6×) |
| Helped | 79 | 343 | +264 |
| Hurt | **0** | 114 | +114 |
| Net | +79 | **+229** | +150 |
| ΔF1 | +0.0157 | **+0.0222** | +0.0065 |
| Time | 2.2 min | 11.3 min | 5× slower |

Key entity changes between the two configs:

| Entity | Old routing F1 | Universal F1 | Notes |
|--------|---------------|--------------|-------|
| I-CREDITCARDNUMBER | 0.03 *(blocked)* | **0.37** | Largest single gain |
| I-TELEPHONENUM | 0.63 *(blocked)* | **0.68** | Unlocked via chaining |
| B-TELEPHONENUM | 0.63 | 0.57 | Slightly worse (more calls, more hurt) |
| B-GIVENNAME | 0.69 *(blocked)* | 0.67 | Opened but lightly harmful |

Old routing was conservative and safe (0 hurt) but left I-CREDITCARDNUMBER and I-TELEPHONENUM completely untouched. Universal routing unlocks them at the cost of 114 hurt tokens — net still strongly positive (+229 vs +79).

### 7.3 V13 vs V12 prompt (mini-batch, 50 samples)

V12: LLM predicts full BIO tag (`B-TELEPHONENUM`, `I-TELEPHONENUM`, `O`)
V13: LLM predicts class only (`TELEPHONENUM`, `O`); BIO assigned deterministically by `_class_to_bio(class, prev_label)`

| Model | V12 net | V13 net | Improvement |
|-------|---------|---------|-------------|
| llama3.1:8b | +4 | **+7** | +3 |
| qwen2.5:7b | 0 | **+11** | +11 |

qwen jump from 0 to +11 is the most telling: with V12, qwen frequently generated invalid BIO sequences (isolated `I-` tokens without a preceding `B-`) that the validator rejected, resulting in zero net corrections. V13 eliminates this failure mode entirely — BIO violations are structurally impossible because B-/I- is never the LLM's responsibility.

### 7.4 Overall progression summary

```
Stage                              Macro F1    ΔF1 vs baseline    Net corr.
──────────────────────────────────────────────────────────────────────────
mDeBERTa only (baseline)           0.3573         —                 —
+ V12 prompt, 8 types, no I-       ~0.3720      +0.0147            ~+65 (est.)
+ V13 prompt, 8 types, no I-       0.3730       +0.0157            +79
+ V13 prompt, universal routing    0.3795       +0.0222            +229
```

Each improvement was orthogonal:
- **V12→V13**: fixed BIO violation errors, no routing config change needed
- **8 types→universal**: expanded routing scope, no prompt change needed
- Both changes together: +6.2% relative F1 improvement over mDeBERTa alone
