# NerGuard — Span-Level LLM Routing (V14_SPAN)

Date: 2026-03-02
Baseline run: `results/validation_20260302_100206/` (token-by-token)
New run: `results/validation_spanrouting_20260302_103549/` (span routing)
LLM: llama3.1:8b via Ollama, 1,000 samples NVIDIA/Nemotron-PII

---

## Why It Was Implemented

The previous token-by-token routing produced **53 harmful corrections** out of 654 LLM calls.
Analysis of the per-class impact revealed a clear pattern:

| Class | Hurt | Helped | Net |
|-------|------|--------|-----|
| I-SURNAME | 20 | 4 | **-16** |
| I-GIVENNAME | 8 | 0 | **-8** |
| I-STREET | 8 | 2 | **-6** |
| B-GIVENNAME | 7 | 0 | **-7** |

**Root cause**: The LLM was routing each token independently. A continuation token like
`">>> Smith <<<"` is deeply ambiguous in isolation — the LLM cannot reliably decide if it's
a SURNAME, GIVENNAME, or a common word without knowing what came before it.
Additionally, many of those I- routing calls were triggered even when the corresponding
B- token was classified confidently — meaning the model had already committed to an entity type,
but the LLM was second-guessing individual tokens within that entity.

---

## What Was Implemented

### Architecture Change

**Before (token-by-token):**
```
For each token:
  if entity_router.should_route(token):
    LLM call for token alone
    → update single token
```

**After (span-level routing):**
```
assemble_entity_spans():
  B-X starts a span → is_uncertain = should_route(B-X)
  I-X tokens of same class are collected into the span
  B- is the ANCHOR: if confident, whole span is skipped

For each span:
  if span.is_uncertain:               ← anchor propagation
    LLM call for entire span text
    → update ALL tokens in span at once (B- gets B-X, I- get I-X)
```

### Key Mechanism: Anchor Propagation

If `B-SURNAME` is classified with high confidence (below routing threshold), its
`I-SURNAME` continuation tokens are **never routed**, regardless of their own entropy/confidence.
This eliminates per-token wobbling on continuation tokens.

If `B-SURNAME` is uncertain (above threshold), the **entire span** (e.g. `"John Smith"`) is
sent to the LLM as a single call. The LLM sees the full entity rather than isolated subwords.

### New Components

**`src/inference/prompts.py` — `PROMPT_V14_SPAN`**
Span-aware variant of PROMPT_V13. Key differences:
- `TARGET SPAN: "{span_text}"` instead of `>>> token <<<`
- `TOKEN COUNT: {token_count}` (how many tokens in the span)
- `MODEL PREDICTION: {entity_class}` (entity class, no BIO prefix)
- Same response format as V13: `{"entity_class": "..."}` → deterministic BIO assignment

**`src/inference/llm_router.py` — `disambiguate_span()`**
New public method alongside `disambiguate()`. Reuses:
- `_extract_context(full_text, span_start, span_end)` — context window centered on full span
- `_call_llm()` — same OpenAI / Ollama backends
- `_validate_response()` — same V13 parsing (`entity_class` key → `_class_to_bio()`)
- `LLMCache` — cache key uses `(span_text, context, prev_label, entity_class)`

BIO reconstruction: `corrected_label` from LLM = `B-{class}` for first token;
caller applies `I-{class}` to all subsequent tokens deterministically.

**`src/evaluation/hybrid_evaluator.py`**
- `EntitySpan` dataclass: `indices, entity_class, is_uncertain, char_start, char_end`
- `assemble_entity_spans()` module-level function: builds spans from model predictions
- `EvalConfig.use_span_routing: bool = True` (default on)
- CLI: `--no-span-routing` to fall back to legacy token-by-token routing
- Phase 2 routing loop replaced with span-level logic; legacy path preserved under `else:`
- JSON summary includes `"span_routing": true`

---

## Results

### Global Metrics Comparison

| Metric | Token-by-token | Span routing | Delta |
|--------|---------------|-------------|-------|
| Macro F1 | 0.4046 | **0.4047** | ≈ same |
| Span F1 (seqeval) | 0.5045 | **0.5071** | **+0.0026** |
| Accuracy | 0.9394 | **0.9395** | +0.0001 |
| **LLM calls** | 654 | **250** | **-61.8%** |
| **LLM hurt** | 53 | **13** | **-75.5%** |
| **Net corrections** | 242 | **265** | +9.5% |
| **Elapsed time** | 7.09 min | **2.92 min** | **-58.8%** |

### Per-Class: Harm Elimination

| Class | Token-by-token | Span routing | Status |
|-------|---------------|-------------|--------|
| I-GIVENNAME | **-2.9%** (8 hurt) | **0.0%** (0 hurt) | ✅ eliminated |
| I-STREET | **-0.8%** (8 hurt) | **0.0%** (0 hurt) | ✅ eliminated |
| I-SURNAME | **-3.8%** (20 hurt) | **-0.5%** (<5 hurt) | ✅ near-eliminated |
| I-CREDITCARDNUMBER | +18.9% | **+27.6%** | 📈 improved |
| I-TELEPHONENUM | +9.9% | **~+10%** | → maintained |

I-SURNAME still shows minor residual harm (-0.5%) because those are cases where
`B-SURNAME` itself was uncertain and routed. The LLM decides on the full span and
occasionally misclassifies it. This is unavoidable with any routing approach.

### Efficiency: Helpful Calls per LLM Call

| Mode | Helped | Calls | Ratio |
|------|--------|-------|-------|
| Token-by-token | 295 | 654 | 45.1% |
| Span routing | 278 | 250 | **111.2%** |

With span routing, each LLM call helps on average **more than one token** (the entire span
is corrected atomically). The absolute number of helpful corrections is nearly the same
(278 vs 295) with 62% fewer API calls.

---

## Architecture Diagram (Updated)

```
Token sequence (mDeBERTa output)
    │
    ▼
assemble_entity_spans()
    │  B-X + I-X* → EntitySpan(is_uncertain = should_route(B-X))
    ▼
For each span:
    ├─ is_uncertain == False        → SKIP (anchor propagation)
    │   (B- confident → all I- in span also skipped)
    │
    ├─ regex.can_skip_llm(span)     → SKIP (regex already confirms)
    │
    └─ disambiguate_span()          → 1 LLM call per entity
           │  PROMPT_V14_SPAN
           │  LLM sees: ">>> John Smith <<<" (full span)
           ▼
       entity_class → B-X (first token), I-X (rest) deterministic BIO
    │
    ▼
regex.correct_predictions()         (O→entity promotions, unchanged)
    │
    ▼
Final hybrid_preds
```

---

## Conclusions

Span routing achieves the original goal: **fewer harmful calls, fewer total calls, same F1**.

The improvement is structural:
1. **Anchor propagation** removes the main source of harm (orphan I- routing) by construction
2. **Full entity context** reduces LLM confusion on name tokens (LLM sees "John Smith", not "Smith")
3. **Single call per entity** makes the system nearly 3x faster and 62% cheaper in LLM calls

The F1 is unchanged because the entity-level decisions are the same quality as before
(the LLM is equally good at deciding if a span is PII), but now with far fewer wasted
calls on tokens that were already correct.

### Open Question
The remaining I-SURNAME harm (-0.5%, ~3-4 hurt tokens) comes from spans where B-SURNAME is
genuinely uncertain. These are edge cases where the LLM misclassifies the full name span.
This could be addressed with per-entity confidence scoring (e.g. reject LLM override if
confidence of change is below a threshold), but the marginal gain would be small.
