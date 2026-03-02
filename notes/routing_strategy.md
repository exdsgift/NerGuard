# NerGuard — LLM Routing Strategy

## Overview

NerGuard usa un sistema ibrido: un modello NER (mDeBERTa) produce le predizioni base, e un LLM viene invocato selettivamente per correggere i token dove il modello è incerto (alta entropia / bassa confidence). Questa pagina documenta le decisioni di routing derivate da analisi empirica su 1.000 campioni del dataset NVIDIA/Nemotron-PII.

---

## Il paradigma V13 (class-only)

Prima di V13, l'LLM doveva predire il BIO label completo (`B-TELEPHONENUM`, `I-TELEPHONENUM`, ecc.). Questo causava frequenti violazioni BIO sui modelli OSS:

- **qwen2.5:7b**: 21/26 errori erano BIO REJECT
- **llama3.1:8b**: 13/16 errori erano BIO REJECT

Con **V13**, l'LLM predice solo la **classe semantica** (`TELEPHONENUM`, `O`, ecc.) e il prefisso B-/I- viene assegnato deterministicamente in base al label del token precedente:

```python
def _class_to_bio(entity_class: str, prev_label: str) -> str:
    if entity_class == "O":
        return "O"
    prev_type = prev_label.replace("B-", "").replace("I-", "") if prev_label != "O" else ""
    return f"I-{entity_class}" if prev_type == entity_class else f"B-{entity_class}"
```

Questo elimina tutti gli errori BIO per costruzione.

---

## Analisi empirica del routing

### Configurazione di riferimento (baseline routing)

8 entità routable, nessun token I-, 1.000 campioni, llama3.1:8b:

| Metrica | Valore |
|---------|--------|
| LLM calls | 172 |
| Helped | 79 |
| Hurt | 0 |
| Net | +79 |
| ΔF1 macro | +0.0157 |
| Tempo | 2.2 min |

### Esperimento 1: routing universale (tutte le entità + tutti I-)

20 entità routable (tutte), token I- sbloccati per tutte, 1.000 campioni, llama3.1:8b:

| Metrica | Valore |
|---------|--------|
| LLM calls | 1.047 |
| Helped | 343 |
| Hurt | 114 |
| Net | +229 |
| ΔF1 macro | +0.0222 |
| Tempo | 11.3 min |

### Breakdown per classe — Esperimento 1

| Label | Istanze | Base acc | Hybrid acc | Delta | Calls | Helped | Hurt | Net |
|-------|---------|----------|------------|-------|-------|--------|------|-----|
| **I-CREDITCARDNUMBER** | 540 | 1.5% | 23.7% | **+22.2%** | 152 | 120 | 0 | **+120** |
| **I-TELEPHONENUM** | 1.247 | 61.7% | 67.8% | **+6.1%** | 317 | 109 | 33 | **+76** |
| B-TELEPHONENUM | 356 | 45.2% | 57.3% | +12.1% | 93 | 43 | 0 | +43 |
| B-CREDITCARDNUMBER | 89 | 2.2% | 38.2% | +36.0% | 35 | 32 | 0 | +32 |
| B-DATE | 961 | 79.1% | 80.2% | +1.1% | 42 | 11 | 0 | +11 |
| B-SURNAME | 523 | 54.7% | 56.6% | +1.9% | 77 | 10 | 0 | +10 |
| B-GENDER | 70 | 1.4% | 7.1% | +5.7% | 5 | 4 | 0 | +4 |
| B-TAXNUM | 22 | 27.3% | 31.8% | +4.5% | 3 | 1 | 0 | +1 |
| I-EMAIL | 2.727 | 98.5% | 98.2% | -0.2% | 15 | 0 | 6 | **-6** |
| I-STREET | 734 | 68.1% | 67.3% | -0.8% | 32 | 2 | 8 | **-6** |
| B-GIVENNAME | 778 | 58.2% | 57.3% | -0.9% | 53 | 0 | 7 | **-7** |
| I-DATE | 1.910 | 84.7% | 84.0% | -0.6% | 63 | 6 | 18 | **-12** |
| I-GIVENNAME | 564 | 41.1% | 38.5% | -2.7% | 66 | 0 | 15 | **-15** |
| I-SURNAME | 424 | 64.6% | 60.8% | -3.8% | 53 | 4 | 20 | **-16** |

---

## Interpretazione

### Perché i token I- numerici beneficiano del routing

Per entità come `CREDITCARDNUMBER` e `TELEPHONENUM`, il contesto rende la classificazione semanticamente semplice: `"4111 >>> 1111 <<< 1111 1111"` è chiaramente una carta di credito indipendentemente dalla posizione nel token span. L'LLM con V13 risponde `CREDITCARDNUMBER` con alta affidabilità, e `_class_to_bio` assegna `I-` correttamente se il token precedente era già parte dello stesso span.

### Perché i token I- testuali vengono danneggiati

Per entità come `SURNAME`, `GIVENNAME`, `DATE`, il token I- fuori contesto è spesso ambiguo: `">>> son <<<"`  in `"Harrison"` potrebbe essere SURNAME, GIVENNAME, o una parola comune. L'LLM commette errori di classificazione del tipo (`GIVENNAME` vs `SURNAME` vs `O`) che deteriorano il risultato. Stesso problema per `I-DATE` dove il modello confonde date-parts con altri numeri.

### Pattern generale

| Categoria I- | V13 sicuro? | Motivo |
|---|---|---|
| **Numerico** (CREDITCARDNUMBER, TELEPHONENUM) | ✓ Sì | Classe ovvia dal pattern numerico nel contesto |
| **Testuale nome** (GIVENNAME, SURNAME) | ✗ No | Token ambigui, LLM commette errori di tipo |
| **Testuale data** (DATE) | ✗ No | Confusione date-parts / numeri generici |
| **Sequenze dense** (EMAIL, STREET) | ✗ No | Alta accuracy baseline, routing peggiora |

---

## Configurazione ottimale (derivata empiricamente)

### Regola di routing B- token

Routare tutti i B- token incerti **eccetto GIVENNAME** (netto -7 nel test, l'LLM confonde con altri nomi o classi affini).

```python
ROUTABLE_ENTITIES = ENTITY_CLASSES - {"GIVENNAME"}
BLOCKED_ENTITIES  = {"GIVENNAME"}
```

### Regola di routing I- token

Routare i token I- **solo** per `CREDITCARDNUMBER` e `TELEPHONENUM`. Tutte le altre entità rimangono bloccate sui token I-.

```python
ROUTABLE_I_ENTITIES = {"CREDITCARDNUMBER", "TELEPHONENUM"}
```

La logica in `entity_router.should_route`:

```python
if block_continuation_tokens and predicted_label.startswith("I-"):
    if entity_type not in routable_i_entities:
        return False  # blocca
    # solo CREDITCARDNUMBER e TELEPHONENUM passano
```

### Stima net corrections con configurazione ottimale

Sommando le contribuzioni positive e sottraendo le negative evitate:

| Fonte | Net stimato |
|-------|------------|
| I-CREDITCARDNUMBER (aperto) | +120 |
| I-TELEPHONENUM (aperto) | +76 |
| B-TELEPHONENUM | +43 |
| B-CREDITCARDNUMBER | +32 |
| B-DATE | +11 |
| B-SURNAME | +10 |
| B-GENDER | +4 |
| B-TAXNUM | +1 |
| B-GIVENNAME (bloccato) | 0 (era -7) |
| I-SURNAME (bloccato) | 0 (era -16) |
| I-GIVENNAME (bloccato) | 0 (era -15) |
| I-DATE (bloccato) | 0 (era -12) |
| I-EMAIL (bloccato) | 0 (era -6) |
| I-STREET (bloccato) | 0 (era -6) |
| **Totale stimato** | **~+297** |

Con molte meno chiamate LLM rispetto al routing universale (~400 vs ~1.047).

---

## Modelli testati

| Modello | Net | ΔF1 | Hurt | Tempo |
|---------|-----|-----|------|-------|
| gpt-oss:20b | +89 | +0.0162 | 3 | 8.6 min |
| llama3.1:8b | +79 | +0.0157 | 0 | 2.2 min |
| qwen2.5:7b | +73 | +0.0141 | 6 | 1.9 min |

> **llama3.1:8b** è il modello OSS raccomandato: 0 hurt, quasi stesso risultato di gpt-oss, 4× più veloce.

---

## File chiave

| File | Ruolo |
|------|-------|
| `src/core/constants.py` | `ROUTABLE_ENTITIES`, `BLOCKED_ENTITIES`, `ROUTABLE_I_ENTITIES` |
| `src/inference/entity_router.py` | Logica di routing: soglie, blocco I-, filtro per tipo |
| `src/inference/llm_router.py` | Chiamata LLM, validazione risposta, `_class_to_bio()` |
| `src/inference/prompts.py` | `PROMPT_V13` (class-only), `PROMPT_V9` (OpenAI full BIO) |
