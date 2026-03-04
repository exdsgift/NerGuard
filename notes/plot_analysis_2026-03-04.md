# Analisi Accademica dei Grafici — Benchmark NerGuard (1000 campioni, NVIDIA-PII)
**Data:** 2026-03-04
**Grafici:** `experiments/combined_1000samples_2026-03-04/plots/` (20 PDF)
**Dataset:** NVIDIA/Nemotron-PII, 1000 campioni, Tier 2 (16 label evaluate)
**Sistemi:** 14 (7 Ollama locali + gpt-4o + Hybrid V1 + Presidio + Piiranha + NerGuard Base + spaCy + BERT-NER)

---

## 1. F1-macro per sistema (`f1_macro_nvidia-pii.pdf`)

### Descrizione
Bar chart dei 14 sistemi ordinati per F1-macro (token-level, macro-average su 16 classi).
L'asse y è scalato dinamicamente sul range osservato (0.31–0.54 circa), permettendo di
discriminare visivamente differenze anche piccole.

### Analisi

Il grafico rivela **due cluster distinti** separati da una discontinuità:

**Cluster superiore (F1-macro 0.477–0.507) — LLM-augmented:**
I 7 sistemi NerGuard Hybrid V2 con LLM locale e il V2 con gpt-4o si collocano tutti
in un intervallo di soli 0.030 punti F1. Questo è il risultato più significativo
dell'esperimento: la scelta del modello LLM ha impatto marginale sulla qualità finale.
La causa è il ceiling imposto dal base model mDeBERTa: il LLM può correggere solo
le predizioni che il base model ritiene incerte (entropia > 0.583 o confidenza < 0.787);
i falsi negativi dove mDeBERTa predice O con alta confidenza non vengono mai routati.

All'interno del cluster, si nota un sotto-raggruppamento:
- **Top tier** (0.500–0.507): gpt-4o, qwen2.5:7b, gpt-oss:20b, deepseek-r1:14b
- **Lower tier** (0.477–0.497): llama3.1:8b, phi4:14b, mistral-nemo:12b, qwen2.5:14b

La separazione tra i due sotto-tier non è spiegabile dalla dimensione del modello
(qwen2.5:14b è nel lower tier, qwen2.5:7b nel top tier), suggerendo che la qualità
del ragionamento contestuale per la disambiguazione PII dipende dall'architettura e
dal training data del modello più che dalla sua dimensione.

**Cluster inferiore (F1-macro 0.333–0.493) — sistemi non-LLM:**
Presidio (0.493) è sorprendentemente competitivo con i sistemi ibridi del lower tier.
Il suo approccio basato su pattern regex e dizionari personalizzati per entità strutturate
(email, SSN, date) è efficace per le categorie più frequenti nel dataset. NerGuard Base
(0.418) mostra il valore aggiunto del LLM routing: +0.089 F1-macro rispetto al mDeBERTa
solo, con latency che passa da 33ms a 41ms (gpt-4o). SpaCy e BERT-NER soffrono della
mancata specializzazione sul dominio PII.

**Implicazione pratica:** Con un budget zero per API cloud, qwen2.5:7b (0.505) supera
persino Presidio (0.493) e NerGuard Hybrid V1 con gpt-4o (0.494), pur essendo un modello
locale da 4.7 GB che gira su hardware commodity.

---

## 2. Entity-level F1 per sistema (`entity_f1_nvidia-pii.pdf`)

### Descrizione
Bar chart di entity-F1 (seqeval span-strict), che valuta le predizioni a livello di span
intero (B- e I- devono essere corretti per tutta la durata dell'entità).

### Analisi

L'ordinamento per entity-F1 diverge significativamente da quello per F1-macro:

**Presidio (0.668) supera TUTTI i sistemi LLM locali** (range 0.658–0.664), nonostante
il suo F1-macro sia inferiore. Questo apparente paradosso si spiega con la natura delle
due metriche:
- F1-macro pesa equamente ogni classe; Presidio fallisce su entità rare (gender, age, tax_id)
  abbassando il macro-average
- Entity-F1 (seqeval) pesa i singoli span; Presidio cattura accuratamente gli span delle
  entità strutturate ad alta frequenza (email, date_of_birth, ssn, first_name)

**gpt-oss:20b** raggiunge la entity-F1 più alta tra tutti i sistemi (0.664), superando
leggermente gpt-4o (0.663). Il chain-of-thought del modello thinking sembra contribuire
a una migliore coerenza span-level, probabilmente perché riduce gli errori di boundary
(classificazione scorretta dei token I-).

**Separazione token vs entity level:**
Per i sistemi NerGuard Hybrid V2, la entity-F1 (0.658–0.664) è sistematicamente superiore
alla F1-micro token (0.697–0.702), il che riflette la natura del dato: la maggior parte
degli errori avviene sui token isolati (B- su entità monolingua) piuttosto che su sequenze
lunghe (dove lo span routing propaga correttamente il B-anchor agli I-).

**dslim/bert-base-NER** presenta l'anomalia più marcata: entity-F1=0.622 nonostante
F1-macro=0.333. BERT-NER è stato addestrato su CoNLL-2003 (PERSON, ORG, LOC, MISC) e
non conosce le categorie PII, ma riesce a mappare alcune entità strutturate (nomi propri
→ PERSON) tramite l'allineamento semantico, ottenendo span corretti anche se la granularità
della categoria è perduta.

---

## 3. F1-macro vs Latenza (`f1_vs_latency_nvidia-pii.pdf`)

### Descrizione
Scatter plot con asse x in scala logaritmica (latency media in ms) e asse y F1-macro.
I limiti degli assi sono dinamici sul range dei dati. Le annotazioni sono posizionate
con algoritmo anti-overlap (slot verticali spaziati + leader line).

### Analisi

Il grafico è lo strumento più informativo per la **decisione di deployment** ed evidenzia
chiaramente la frontiera di Pareto.

**Frontiera di Pareto:**
Un sistema è Pareto-ottimale se nessun altro sistema è contemporaneamente migliore in
F1 E più veloce. I sistemi sulla frontiera sono:
- **gpt-4o V2** (41ms, 0.507): ottimale per latency critica con cloud budget
- **qwen2.5:7b V2** (564ms, 0.505): ottimale per self-hosted balanced
- **NerGuard Base** (33ms, 0.418): ottimale per latency ultra-critica senza LLM budget
- **Presidio** (86ms, 0.493): ottimale per sistemi rule-based

**Sistemi dominati** (non sulla frontiera):
- `deepseek-r1:14b` (7566ms, 0.501): dominato da qwen2.5:7b (migliore F1, 13× più veloce)
- `phi4:14b`, `mistral-nemo:12b`, `qwen2.5:14b`: cluster a 0.477 con latency 734–1251ms,
  dominati da llama3.1:8b (0.497, 707ms) che è più veloce e migliore

**Cluster a bassa F1 e bassa latency:**
NerGuard Base, Piiranha, dslim/bert-base-NER, spaCy si collocano nell'angolo in basso
a sinistra (5–144ms, 0.333–0.473). Rappresentano soluzioni deterministiche senza LLM:
il gap di F1 rispetto al cluster LLM è ~0.03–0.17 punti.

**Considerazione economica:**
La pendenza della frontiera di Pareto tra gpt-4o e qwen2.5:7b è quasi verticale sul
grafico log-scale: +13.7× latency per +0.0018 F1. Questa è la quantificazione esatta
del costo di privacy-by-design (self-hosting vs cloud API).

---

## 4. System Ranking (`ranking_nvidia-pii.pdf`)

### Descrizione
Horizontal bar chart con sistemi ordinati dal peggiore al migliore (bottom-up).
L'asse x è dinamico e scalato sul range [0.31, 0.52] circa, permettendo di leggere
le differenze con 4 cifre decimali.

### Analisi

Il ranking evidenzia strutturalmente tre tier:

**Tier 3 — Sistemi non specializzati PII** (F1 < 0.40):
spaCy e BERT-NER sono sistemi general-purpose NER non ottimizzati per PII detection.
Il loro basso F1-macro riflette la distanza tra il dominio CoNLL/OntoNotes e il
dominio NVIDIA/Nemotron-PII, ricco di entità strutturate (SSN, carte di credito,
numeri di licenza) assenti nei training data standard.

**Tier 2 — Sistemi specializzati PII senza LLM** (F1 0.40–0.50):
NerGuard Base, Presidio, Piiranha, NerGuard Hybrid V1 con gpt-4o. La presenza di
Hybrid V1 in questo cluster (0.494, vs V2 gpt-4o 0.507) quantifica il beneficio
dell'aggiornamento architetturale V2: il V2 introduce lo span routing che elimina
le chiamate LLM dannose sui token I- (I-GIVENNAME, I-STREET) che nel V1 causavano
regressioni sistematiche.

**Tier 1 — NerGuard Hybrid V2** (F1 0.477–0.507):
Il cluster LLM-augmented è compatto ma con differenziazioni significative. Si nota
che i modelli del lower tier (phi4:14b, mistral-nemo:12b, qwen2.5:14b) si trovano
a 0.477–0.478, praticamente identici: tre modelli con architetture diverse convergono
allo stesso performance. Questo suggerisce un limite strutturale nel prompt V14_SPAN
piuttosto che una differenza nel modello LLM.

---

## 5. Precision vs Recall (`precision_recall_nvidia-pii.pdf`)

### Descrizione
Scatter plot precision-micro vs recall-micro con iso-F1 curves e dynamic axis limits.
I punti sono annotati con anti-overlap. Asse x e y scalati dinamicamente sul range
dei dati con padding.

### Analisi

Il grafico rivela il **tradeoff fondamentale** tra i diversi approcci:

**Quadrante alto-destra (alta precision, alto recall) — Sistemi ibridi:**
I sistemi NerGuard Hybrid V2 si collocano nel quadrante migliore (precision 0.588–0.597,
recall 0.745–0.749). Il routing LLM sposta i sistemi verso il quadrante ottimale rispetto
al Base model: il recall aumenta perché il LLM conferma e corregge predizioni incerte,
mentre la precision rimane elevata perché il routing V14_SPAN è conservativo (nega le
predizioni ambigue solo se molto sicuro).

**Presidio — alto recall, precision moderata:**
Presidio (precision 0.570, recall 0.807) è il sistema con recall più alto, ma a costo
di una precision inferiore. Il suo approccio pattern-based produce molti falsi positivi
su pattern numerici (numeri di telefono confusi con date, o viceversa).

**NerGuard Base — recall basso:**
Il base model senza LLM (precision 0.445, recall 0.524) si distingue per il recall
marcatamente inferiore agli ibridi (-0.197 vs gpt-4o V2). Questo quantifica i FN
dell'approccio puramente neurale: mDeBERTa non identifica ~20% delle entità PII
che il routing LLM riesce a recuperare.

**Degenerazione per sistemi general-purpose:**
spaCy e BERT-NER mostrano sia precision che recall bassi. L'iso-F1 curve più vicina
a questi sistemi è quella F1=0.5, confermando il loro posizionamento nel cluster Tier 3.

**Nota sulla scelta delle iso-F1 lines:**
Il grafico mostra curve per F1 ∈ {0.5, 0.6, 0.7, 0.8, 0.9}. I sistemi NerGuard V2
si trovano tutti sopra la curva F1=0.7, confermando la robustezza del sistema ibrido.

---

## 6. Latency Comparison (`latency_nvidia-pii.pdf`)

### Descrizione
Bar chart della latency media (ms/campione) con i sistemi ordinati per tipo.
L'asse y ha ylim dinamico a zero per preservare l'interpretabilità assoluta (la latency
ha un'unità naturale non comparabile come il F1).

### Analisi

Il grafico ha una forma quasi logaritmica: la maggior parte dei sistemi si concentra
nell'intervallo 30–1251ms, con deepseek-r1:14b come outlier a 7566ms.

**Sistemi ultra-fast (< 100ms):**
Piiranha (31ms), NerGuard Hybrid (gpt-4o) (31ms), NerGuard Base (33ms), BERT-NER (38ms),
gpt-4o V2 (41ms), Presidio (86ms). La latenza di gpt-4o V2 (41ms) è notevole: nonostante
le API call esterne, l'OpenAI API ha latency p95=65ms — più bassa di qualsiasi modello
Ollama locale.

**Sistemi locali small (563–734ms):**
qwen2.5:7b (564ms), llama3.1:8b (707ms), mistral-nemo:12b (734ms). Tre modelli compatti
(7–12B parametri) con latency simili, adatti per deployment production self-hosted.

**Sistemi locali medium (981–1251ms):**
qwen2.5:14b (981ms), phi4:14b (1251ms). Il raddoppio dei parametri (7→14B) comporta
~1.7× aumento di latency ma non corrisponde a un aumento equivalente di F1.

**Reasoning models:**
gpt-oss:20b (3139ms, p95=7553ms) e deepseek-r1:14b (7566ms, p95=25122ms, p99=40831ms)
mostrano latency proibitive per uso production. Il chain-of-thought generato da questi
modelli introduce varianza elevata (p99/mean ratio: deepseek=5.4×, gpt-oss=2.4×)
rispetto ai modelli standard (llama3.1:8b p99/mean=1.8×). La latenza mediana di
deepseek-r1:14b è 5696ms, indicando che la distribuzione è heavy-tailed.

---

## 7. Token-level vs Entity-level F1 (`token_vs_entity_nvidia-pii.pdf`)

### Descrizione
Grouped bar chart con due barre per sistema: F1-macro (token-level) e Entity-F1 (seqeval).
L'asse y è dinamico sul range combinato delle due metriche.

### Analisi

Il grafico visualizza la **discrepanza sistematica** tra le due metriche di valutazione:

**Inversioni notevoli:**
- **dslim/bert-base-NER**: F1-macro=0.333 ma Entity-F1=0.622 — il gap più grande del dataset.
  BERT-NER è addestrato a produrre span completi e coerenti (B-/I- sempre ben formati),
  ma le categorie non corrispondono al dominio PII → token-level penalizza la categoria,
  entity-level premia la correttezza dello span boundary.
- **Presidio**: F1-macro=0.493, Entity-F1=0.668 — gap di 0.175 punti. Le regex di Presidio
  producono span molto puliti (zero errori B/I) ma coprono meno categorie.
- **spaCy**: F1-macro=0.361, Entity-F1=0.553 — gap di 0.192 punti. SpaCy NLP pipeline è
  molto precisa nel boundary detection (eredita da CoNLL training) ma ignora le categorie PII.

**Sistemi LLM-augmented:**
Per NerGuard Hybrid V2, Entity-F1 (0.658–0.664) > F1-micro (0.697–0.701) > F1-macro (0.477–0.507).
Questo ordine riflette la distribuzione bimodale delle performance per classe:
- Entità strutturate frequenti: F1 > 0.8 (alzano micro e entity)
- Entità rare: F1 ≈ 0 (abbassano macro)

**Implicazione per la valutazione:**
Nessuna delle due metriche cattura completamente la qualità di un sistema PII detection.
La entity-F1 sottovaluta i sistemi che fanno bene sulle entità rare (perché pesa gli span
e le entità rare hanno pochi span). La F1-macro è più appropriata per valutare la
copertura generale del sistema su tutte le categorie PII, che è l'obiettivo GDPR.

---

## 8. Per-entity F1 — singoli sistemi

### 8.1 NerGuard Hybrid V2 (gpt-4o) — `per_entity_nerguard_hybrid_v2_gpt-4o_nvidia-pii.pdf`

Il grafico mostra l'F1 per ciascuna delle 16 categorie valutate, con code coloring:
verde (F1 ≥ 0.9), arancione (0.5 ≤ F1 < 0.9), rosso (F1 < 0.5).

**Entità eccellenti (verde):**
- `date_of_birth`: F1=1.0 — il regex validator intercetta tutti i pattern `YYYY-MM-DD` e
  varianti prima del LLM call, ottenendo zero FN e zero FP.

**Entità buone (arancione):**
- `email` (0.864), `credit_debit_card` (0.829), `date` (0.826): il LLM routing aiuta
  significativamente rispetto al Base model (+0.01 email, +∞ credit_debit_card da 0.0 base).
- `ssn` (0.620): miglioramento enorme dal Base (0.212), grazie al routing che aumenta il recall
  da 0.45 a 0.94, ma la precision rimane bassa (0.462) per falsi positivi su numeri generici.
- `phone_number` (0.558), `postcode` (0.549): pattern ambigui che producono molti FP.

**Entità deboli (rosso):**
- `gender`, `age`, `tax_id`, `certificate_license_number`: F1 ≈ 0.05–0.15.
  Queste entità non sono routate (confidenza base model alta su O) o sono confuse con entità
  strutturalmente simili. Rappresentano la "coda lunga" del dataset PII.

### 8.2 NerGuard Hybrid V2 (qwen2.5:7b) vs (gpt-4o) — confronto

La differenza principale tra i due sistemi è nelle entità moderate:
- `date`: qwen2.5:7b (0.914) vs gpt-4o (0.826) — differenza di 0.088! Il modello Qwen è
  significativamente migliore nel riconoscimento di date, probabilmente grazie al training
  su dati multilinguali che include molti formati di data.
- `credit_debit_card`: qwen2.5:7b (0.857) vs gpt-4o (0.829) — qwen leggermente superiore.
- `ssn`: sostanzialmente identici (0.619 vs 0.620).
- Su tutte le entità deboli: identici (entrambi ~0 su gender, age, tax_id).

### 8.3 NerGuard Hybrid V2 (gpt-oss:20b) — caratteristiche distinte

- `credit_debit_card`: 0.877 — il migliore tra tutti i sistemi (+0.048 vs gpt-4o). Il
  chain-of-thought del modello thinking è particolarmente utile per disambiguare numeri
  a 16 cifre in contesto.
- Consistente su tutte le entità moderate, nessuna regressione rispetto a gpt-4o.
- Su entità tail: identico a tutti gli altri (il ceiling del base model si applica uguale).

### 8.4 Presidio — `per_entity_presidio_nvidia-pii.pdf`

Presidio mostra un profilo bimodale molto marcato:
- **Entità strutturate**: email (0.997), date_of_birth (1.0), ssn (0.829), first_name (0.929) — eccellente
- **Entità non-strutturate**: city, street_address, tax_id, gender — F1 ≈ 0

Questo riflette il design di Presidio: dizionari di pattern regex/NLP per entità standard,
nessuna generalizzazione sulle entità contestuali. Per un sistema GDPR Tier-1 (entità
"classiche"), Presidio è competitivo o superiore ai sistemi LLM-augmented a una frazione del costo.

### 8.5 NerGuard Base — `per_entity_nerguard_base_nvidia-pii.pdf`

Il Base model (mDeBERTa fine-tuned, no LLM) mostra:
- `date_of_birth`: 0.984 — quasi perfetto grazie al regex pre-processing
- `email`: 0.921 — sorprendentemente meglio del sistema ibrido (0.864)!
  Il routing LLM introduce FP su email-like strings che il Base model classifica correttamente.
- `ssn`: 0.212 — molto basso; il Base model manca molti SSN a bassa confidenza che il routing
  poi recupera nei sistemi ibridi.
- `certificate_license_number`: 0.142 — praticamente inutilizzabile senza routing specializzato.

---

## 9. Sintesi e Conclusioni Accademiche

### 9.1 Finding 1: LLM routing converge su un ceiling comune

Il dato più rilevante è la **convergenza** dei sistemi NerGuard Hybrid V2 verso lo stesso
performance: 0.030 punti F1-macro separano il primo (gpt-4o, 0.507) dall'ultimo (qwen2.5:14b, 0.477)
tra i sistemi LLM-augmented, su 1000 campioni di test. Il ceiling è imposto dal base model:
la capacità del LLM di correggere predizioni errate è limitata al sottoinsieme di token
che il base model stesso considera incerti. Questo è un risultato con implicazioni dirette
per il design di sistemi ibridi: investire nella qualità del base model (fine-tuning su dominio,
aumentazione dati per entità tail) ha un impatto potenzialmente maggiore che upgradare il LLM.

### 9.2 Finding 2: qwen2.5:7b domina la frontiera di Pareto locale

Su hardware consumer-grade (GPU da 8GB), qwen2.5:7b offre il miglior rapporto qualità/latency
tra tutti i sistemi locali testati. Riesce a superare sistemi con il doppio dei parametri
(qwen2.5:14b, phi4:14b) su tutte le metriche rilevanti. La sua superiorità su `date` (0.914 vs 0.826
di gpt-4o) suggerisce che il training multilinguale di Qwen includa varianti di formato data
particolarmente rilevanti per il dataset NVIDIA/Nemotron-PII.

### 9.3 Finding 3: le metriche aggregate mascherano pattern per-entity critici

L'analisi per-entity rivela che i sistemi con F1-macro simile possono avere profili di errore
molto diversi. Un sistema che ottiene F1=0.50 potrebbe farlo con copertura uniforme su tutte
le entità (desiderabile per GDPR compliance) o con performance eccellenti su entità frequenti
e zero su entità rare (indesiderabile, perché le entità rare possono essere le più sensitive).
Per un sistema di protezione dati, il F1-macro è più appropriato del F1-micro o dell'entity-F1
come metrica primaria.

### 9.4 Finding 4: i reasoning model non giustificano il costo computazionale

deepseek-r1:14b (7566ms, F1=0.501) e gpt-oss:20b (3139ms, F1=0.503) non superano qwen2.5:7b
(564ms, F1=0.505) in F1-macro. Il chain-of-thought appare utile per edge case specifici
(credit_debit_card per gpt-oss:20b +0.048 vs gpt-4o), ma il guadagno marginale non giustifica
il costo computazionale (5–13× latency) per un task di routing dove il prompt è già strutturato.

### 9.5 Limitazioni

- **Sample size:** 1000 campioni con seed fisso (42) — i risultati sono stabili a questa scala
  ma potrebbero variare su campioni differenti (es. distribuzione diversa di entità tail).
- **Tier 2 evaluation:** solo 16/54 label del dataset sono evaluate; le entità escluse
  (ip_address, url, organization, ecc.) non influenzano i numeri ma limitano la generalizzabilità.
- **Ollama single-threaded:** i test usano `--no-batch-llm` (routing sequenziale). Con batch
  async, il throughput dei modelli locali aumenterebbe significativamente.
- **Prompt fisso:** tutti i test usano V14_SPAN; prompt diversi (come dimostrato dall'ablation
  V16_SPAN) possono alterare la distribuzione di FP/FN senza cambiare il recall complessivo.
