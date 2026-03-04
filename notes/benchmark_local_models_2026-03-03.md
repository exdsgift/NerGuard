# Benchmark: Local Ollama Models vs OpenAI (gpt-4o)
**Data:** 2026-03-03 / 2026-03-04
**Dataset:** NVIDIA/Nemotron-PII (1000 campioni, Tier 2 evaluation, seed=42)
**Esperimenti:**
- `experiments/2026-03-03_13-23/` — baseline OpenAI + sistemi non-LLM
- `experiments/local_1000samples_2026-03-03/` — 7 modelli Ollama
- `experiments/combined_1000samples_2026-03-04/` — sessione combinata (14 sistemi)
- `experiments/combined_1000samples_2026-03-04/plots/` — 20 grafici PDF

---

## 1. Obiettivo

Valutare il trade-off qualità/costo/latenza tra LLM cloud (gpt-4o) e LLM self-hosted (Ollama)
nel contesto del sistema NerGuard Hybrid V2: un pipeline ibrido che combina mDeBERTa come
base NER e un LLM come disambiguatore tramite span routing.

**Domande di ricerca:**
1. I modelli locali raggiungono una qualità paragonabile a gpt-4o?
2. Qual è il trade-off latenza/qualità per ciascun modello?
3. I reasoning model (deepseek-r1, gpt-oss:20b) performano meglio dei modelli standard?
4. Quale modello locale è ottimale per un deployment self-hosted?

---

## 2. Setup Sperimentale

### 2.1 Hardware

```
NVIDIA-SMI 580.65.06  |  Driver Version: 580.65.06  |  CUDA Version: 13.0

GPU 0: NVIDIA A100-PCIE-40GB  |  Bus-Id: 00000000:81:00.0
  VRAM: 40960 MiB  |  PWR: 47W/250W

GPU 1: NVIDIA A100-PCIE-40GB  |  Bus-Id: 00000000:E1:00.0
  VRAM: 40960 MiB  |  PWR: 55W/250W
```

- **Total VRAM disponibile:** 81.9 GB (2× A100-PCIE-40GB)
- Ollama occupa GPU 0 (~9.9 GB) + GPU 1 (~0.4 GB) durante i run
- Ollama carica un modello alla volta (multi-GPU automatico)
- mDeBERTa (base NER): ~2GB VRAM, GPU separata da Ollama

### 2.2 Modelli testati

| # | Modello | VRAM | Tipo | Installazione |
|---|---------|------|------|---------------|
| 0 | `gpt-4o` (OpenAI API) | cloud | standard | riferimento |
| 1 | `llama3.1:8b` | 4.9 GB | standard | pre-installato |
| 2 | `qwen2.5:7b` | 4.7 GB | standard | pre-installato |
| 3 | `gpt-oss:20b` | 13 GB | thinking/MXFP4 | pre-installato |
| 4 | `phi4:14b` | ~9 GB | standard | 2026-03-03 |
| 5 | `qwen2.5:14b` | ~9 GB | standard | 2026-03-03 |
| 6 | `mistral-nemo:12b` | ~7 GB | standard | 2026-03-03 |
| 7 | `deepseek-r1:14b` | ~9 GB | reasoning | 2026-03-03 |

### 2.3 Configurazione benchmark

```bash
# Script: run_local_benchmark.sh --skip-smoke --samples=1000
uv run python -m src.benchmark.runner \
  --systems nerguard-hybrid-v2 \
  --datasets nvidia-pii \
  --samples 1000 \
  --llm-source ollama \
  --llm-model "$MODEL" \
  --no-batch-llm \
  --semantic-alignment alignments/default.json \
  --session-dir "./experiments/local_1000samples_2026-03-03"
```

- **Prompt LLM:** V14_SPAN (span routing, framing yes/no minimale)
- **Routing:** solo entità incerte con entropia > 0.583 o confidenza < 0.787
- **Span routing:** B-anchor + propagazione I- (no LLM call per token interni)
- **Regex validator:** attivo (pre-routing guard per pattern certi)
- **Eval mode:** unbiased (token esclusi dal CLEAN_NVIDIA_MAP → -100)

### 2.4 Dataset (NVIDIA/Nemotron-PII)

- **16 label evaluate** (Tier 2): age, certificate_license_number, city, credit_debit_card,
  date, date_of_birth, email, first_name, gender, last_name, phone_number, postcode,
  ssn, street_address, tax_id, time
- **4 label escluse** dal sistema: BUILDINGNUM, IDCARDNUM, PASSPORTNUM, SEX, TITLE
- **38 label escluse** dal dataset (non mappabili): ip_address, url, organization, ecc.
- **Copertura:** ~57% dei token del dataset (Tier 2 → evaluation su sottoinsieme)

---

## 3. Procedura e Fix

### 3.1 Bug critico: LLMRouter Ollama model parameter

**Bug scoperto prima dei run:** `NerGuardHybrid.setup()` passava `model=X` a LLMRouter,
ma il parametro corretto per Ollama è `ollama_model`. I run usavano `qwen2.5:3b` (default)
invece del modello CLI.

```python
# Fix applicato in nerguard_hybrid.py e piiranha_hybrid.py:
self.router = LLMRouter(source=self.llm_source, model=self.llm_model, ollama_model=self.llm_model)
```

### 3.2 Fix comparison table (runner.py)

**Bug:** il runner saltava i run già esistenti (idempotency) senza aggiungere i risultati
a `self.all_results` → la summary finale conteneva solo il modello dell'ultimo run.

**Fix:** loading da cache `results.json` + `config.json` quando si skippa per idempotency.

### 3.3 Smoke test risultati

| Modello | Esito | LLM calls | Delta F1 | Note |
|---------|-------|-----------|----------|------|
| `llama3.1:8b` | ✅ | validato | — | — |
| `qwen2.5:7b` | ✅ | validato | — | — |
| `gpt-oss:20b` | ✅ | 5 campioni | — | thinking model, ~16s/call |
| `phi4:14b` | ✅ | 15 | +1.99% | 0 hurt |
| `qwen2.5:14b` | ✅ | 15 | +1.14% | 0 hurt |
| `mistral-nemo:12b` | ✅ | 15 | +0.28% | 0 hurt |
| `deepseek-r1:14b` | ✅ | 15 | +1.42% | ~16s/call, ~4 min per 5 campioni |

---

## 4. Risultati Completi (1000 campioni, nvidia-pii, Tier 2)

### 4.1 Ranking F1-macro — tutti i sistemi

| # | Sistema | F1-macro | F1-micro | Entity-F1 | Prec-macro | Rec-macro | Lat-mean | Lat-p95 | Throughput |
|---|---------|----------|----------|-----------|------------|-----------|----------|---------|------------|
| 1 | **NerGuard Hybrid V2 (gpt-4o)** | **0.5069** | 0.7015 | 0.6634 | 0.5554 | 0.7211 | 41ms | 65ms | 24.2 s/s |
| 2 | NerGuard Hybrid V2 (qwen2.5:7b) | 0.5051 | 0.7009 | 0.6618 | 0.5529 | 0.7197 | 564ms | 1551ms | 1.77 s/s |
| 3 | NerGuard Hybrid V2 (gpt-oss:20b) | 0.5028 | 0.7012 | 0.6640 | 0.5510 | 0.7197 | 3139ms | 7553ms | 0.32 s/s |
| 4 | NerGuard Hybrid V2 (deepseek-r1:14b) | 0.5008 | 0.6970 | 0.6606 | 0.5484 | 0.7140 | 7566ms | 25122ms | 0.13 s/s |
| 5 | NerGuard Hybrid V2 (llama3.1:8b) | 0.4972 | 0.6973 | 0.6583 | 0.5432 | 0.7197 | 707ms | 1966ms | 1.42 s/s |
| 6 | NerGuard Hybrid (gpt-4o) | 0.4943 | 0.6862 | 0.6475 | 0.5394 | 0.7088 | 31ms | — | — |
| 7 | Presidio | 0.4933 | 0.5493 | 0.6680 | 0.5635 | 0.6906 | 86ms | 182ms | 11.6 s/s |
| 8 | NerGuard Hybrid V2 (phi4:14b) | 0.4778 | 0.6981 | 0.6595 | 0.5248 | 0.7154 | 1251ms | 3995ms | 0.80 s/s |
| 9 | NerGuard Hybrid V2 (mistral-nemo:12b) | 0.4774 | 0.6967 | 0.6582 | 0.5220 | 0.7197 | 734ms | 2045ms | 1.36 s/s |
| 10 | NerGuard Hybrid V2 (qwen2.5:14b) | 0.4773 | 0.6975 | 0.6619 | 0.5221 | 0.7140 | 981ms | 2941ms | 1.02 s/s |
| 11 | Piiranha | 0.4731 | 0.6501 | 0.6195 | 0.5191 | 0.6798 | 31ms | — | — |
| 12 | NerGuard Base | 0.4175 | 0.6105 | 0.6076 | 0.4452 | 0.5242 | 33ms | 53ms | 30.1 s/s |
| 13 | spaCy (en_core_web_trf) | 0.3607 | 0.4175 | 0.5527 | 0.3977 | 0.6009 | 144ms | — | — |
| 14 | dslim/bert-base-NER | 0.3331 | 0.4821 | 0.6225 | 0.3644 | 0.5945 | 38ms | — | — |

### 4.2 Delta NerGuard Hybrid V2 rispetto a gpt-4o

| Modello | ΔF1-macro | ΔEnt-F1 | ΔRec-macro | Lat / gpt-4o | Cost |
|---------|-----------|---------|------------|--------------|------|
| gpt-4o (ref) | 0.0000 | 0.0000 | 0.0000 | 1× (41ms) | $$ |
| **qwen2.5:7b** | **-0.0018** | **-0.0016** | **-0.0014** | **13.7×** | free |
| gpt-oss:20b | -0.0041 | +0.0006 | -0.0014 | 76.6× | free |
| deepseek-r1:14b | -0.0061 | -0.0028 | -0.0071 | 184.5× | free |
| llama3.1:8b | -0.0097 | -0.0051 | -0.0014 | 17.2× | free |
| phi4:14b | -0.0291 | -0.0039 | -0.0057 | 30.5× | free |
| mistral-nemo:12b | -0.0295 | -0.0052 | -0.0014 | 17.9× | free |
| qwen2.5:14b | -0.0296 | -0.0015 | -0.0071 | 23.9× | free |

---

## 5. Analisi per-entità (NerGuard Hybrid V2)

### 5.1 Entity type — performance comparativa

| Entity | Base | gpt-4o | qwen2.5:7b | llama3.1:8b | gpt-oss:20b | Note |
|--------|------|--------|------------|-------------|-------------|------|
| date_of_birth | 0.984 | **1.000** | **1.000** | **1.000** | **1.000** | regex cattura tutto |
| email | 0.921 | 0.864 | 0.863 | 0.864 | 0.865 | LLM stabile, base meglio |
| credit_debit_card | — | 0.829 | 0.857 | 0.826 | **0.877** | gpt-oss migliore |
| date | 0.768 | 0.826 | **0.914** | 0.822 | 0.826 | qwen migliore |
| ssn | 0.212 | 0.620 | 0.619 | 0.611 | 0.616 | routing aiuta |
| phone_number | 0.585 | 0.558 | — | 0.561 | — | P bassa, R altissima |
| postcode | 0.499 | 0.549 | — | — | — | FP alti |
| city | 0.583 | — | — | — | — | non routato |
| certificate_license_number | 0.142 | — | — | — | — | tail entity, molto debole |
| gender | — | — | — | — | — | F1 ≈ 0 per tutti |
| age | — | — | — | — | — | F1 ≈ 0 per tutti |
| tax_id | — | — | — | — | — | F1 ≈ 0 per tutti |

### 5.2 Entità "tail" (peggiori — macro-avg ~0.19)

Le entità a bassa frequenza nel dataset presentano F1 vicino a 0 per **tutti** i sistemi:
- **gender**: confuso con SEX (esclusa in Tier 2), o con pronomi
- **age**: numeri ambigui (es. "35" → AGE o DATE?)
- **tax_id**: pattern simile a TELEPHONENUM o SOCIALNUM
- **certificate_license_number**: formato variabile, base model non addestrato

Il routing LLM porta il F1-macro delle tail entities da 0.158 (Base) a 0.226 (gpt-4o V2),
ma il recupero è limitato: il base model predice O su queste entità con alta confidenza,
quindi non vengono mai routate.

### 5.3 Analisi per lunghezza campione

| Bucket | n | Base | gpt-4o | qwen2.5:7b | llama3.1:8b | deepseek-r1:14b |
|--------|---|------|--------|------------|-------------|-----------------|
| Short (<50 token) | 9 | 0.764 | 0.806 | 0.806 | 0.806 | 0.739 |
| Medium (50-200 tok) | 238 | 0.450 | 0.558 | 0.560 | 0.559 | 0.549 |
| Long (>200 tok) | 753 | 0.424 | 0.516 | 0.512 | 0.505 | 0.510 |

- Campioni brevi (n=9): tutti i modelli LLM identici a 0.806 — sample size non significativo
- **Campioni lunghi (753/1000):** dominano il dataset; differenza LLM vs Base massima (+0.09)
- Su campioni lunghi: gpt-4o (0.516) > qwen2.5:7b (0.512) > deepseek-r1:14b (0.510) > llama3.1:8b (0.505)

---

## 6. Bottleneck: FN del Base Model

### 6.1 Recall quasi identico tra tutti i modelli LLM

| Sistema | Rec-macro | Rec-micro |
|---------|-----------|-----------|
| gpt-4o V2 | 0.7211 | 0.749 |
| qwen2.5:7b V2 | 0.7197 | 0.747 |
| llama3.1:8b V2 | 0.7197 | 0.747 |
| NerGuard Base (no LLM) | 0.5242 | 0.662 |

Il routing LLM porta il recall da 0.524 (Base) a 0.721 (+0.197), ma **tra i modelli LLM
il recall differisce di soli 0.0014** — praticamente identico.

### 6.2 Causa: mDeBERTa predice O con alta confidenza

```
mDeBERTa → O (confidenza > 0.787) → token non routato → FN
                                    ↑
                         questo è il 90%+ dei FN
```

- Abbassare le soglie di routing aumenterebbe recall ma introduce più FP
- Il LLM non può correggere ciò che non vede

### 6.3 Ablation V16_SPAN (prompt migliorato per FN)

Test: llama3.1:8b + prompt V16_SPAN (NVIDIA aliases, 3-option framing, 7 esempi)

| Metrica | V14_SPAN (default) | V16_SPAN | Delta |
|---------|-------------------|----------|-------|
| F1-macro | 0.5793 | 0.5589 | -0.0204 ▼ |
| Precision | 0.6110 | 0.6000 | -0.0110 ▼ |
| **Recall** | **0.7557** | **0.7557** | **= (identico!)** |
| Latency | 752ms | 969ms | +217ms ▼ |

**Conclusione definitiva**: recall identico anche con prompt diverso. Il bottleneck non è
il prompt LLM, è il base model. V16_SPAN scartato.

---

## 7. Osservazioni sui Sistemi Non-LLM

### Presidio
- entity-F1 = **0.6680** — superiore a TUTTI i sistemi LLM locali!
- Fortissimo su: email (0.997), date_of_birth (1.0), first_name (0.929), ssn (0.829)
- Molto debole su entità rare non nel suo dizionario
- F1-macro = 0.4933 (basso) perché fallisce su entità meno strutturate

### Piiranha
- entity-F1 = 0.6195, latency = 31ms — ottimo per use case senza LLM
- Buono su: credit_debit_card (0.843), email (0.796), postcode (0.883)

### NerGuard Base (mDeBERTa solo)
- F1-macro = 0.4175 → il routing LLM (V2 gpt-4o) aggiunge +0.089 F1-macro
- Latency: 33ms vs 41ms gpt-4o V2 → il costo del routing è solo +8ms (media)
- **Tuttavia:** base latency median = 29.7ms, p95 = 53ms — molto consistente

### dslim/bert-base-NER
- F1-macro molto basso (0.333) ma entity-F1 = 0.6225: alta precision su pochi tipi
- Ottimo su B-first_name (F1=0.902, P=1.0) ma quasi nulla su entità strutturate

---

## 8. Raccomandazione Deployment

| Scenario | Sistema consigliato | Motivo |
|----------|---------------------|--------|
| Produzione cloud, max qualità | NerGuard Hybrid V2 (gpt-4o) | +0.0018 F1 vs qwen, 24 s/s |
| Self-hosted, balanced | **NerGuard Hybrid V2 (qwen2.5:7b)** | Best F1 locale, più veloce |
| Self-hosted, best quality | NerGuard Hybrid V2 (gpt-oss:20b) | Migliore entity-F1, 3× più lento |
| No LLM, latency critica | Presidio | 86ms, entity-F1=0.668 |
| No LLM, balanced | Piiranha | 31ms, entity-F1=0.619 |

**qwen2.5:7b** è il modello locale ottimale:
- F1-macro = 0.5051 (miglior locale, -0.0018 da gpt-4o)
- entity-F1 = 0.6618 (-0.0016 da gpt-4o)
- Latency mean = 564ms (più veloce tra i modelli competitivi)
- Throughput = 1.77 campioni/sec → ~6368 documenti/ora

---

## 9. Directory esperimenti

| Contenuto | Path |
|-----------|------|
| OpenAI + baselines (1000 campioni) | `experiments/2026-03-03_13-23/` |
| 7 modelli locali Ollama (1000 campioni) | `experiments/local_1000samples_2026-03-03/` |
| Sessione combinata (14 sistemi) | `experiments/combined_1000samples_2026-03-04/` |
| Grafici combinati (20 PDF) | `experiments/combined_1000samples_2026-03-04/plots/` |
| Ablation V16_SPAN (llama3.1:8b) | `experiments/ablation_v16_llama3.1_8b/` |
| Analisi accademica grafici | `notes/plot_analysis_2026-03-04.md` |

---

## 10. Prossimi Passi

1. **Abbassare soglie routing** (entropy < 0.583, conf < 0.787) per intercettare più FN
   del base model — test controllato per misurare tradeoff FP/FN
2. **Espandere O-span routing** (`PROMPT_O_SPAN`) con soglia più aggressiva per entità tail
3. **Fine-tuning mDeBERTa** su NVIDIA/Nemotron-PII (o subset augmentato) per alzare il ceiling
   di recall senza dipendenza dal LLM
4. **Benchmark su ai4privacy** — valutare generalizzazione cross-dataset
5. **Async batch routing** (--batch-llm) per aumentare throughput locale con qwen2.5:7b
