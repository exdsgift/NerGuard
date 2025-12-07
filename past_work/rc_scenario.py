"""
Test censura PII su documenti finanziari - Ottimizzato per gretelai/synthetic_pii_finance_multilingual
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import torch
import numpy as np

# ============================================
# CONFIGURAZIONE
# ============================================
MODEL_PATH = "pii_mdeberta_softmax/outputs/best"
CONFIDENCE_THRESHOLD = 0.3  # Abbassato per catturare più entità
REDACTION_CHAR = "█"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carica modello
print(f"📂 Caricamento da: {MODEL_PATH}")
config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE).eval()
id2label = config.id2label

print(f"✅ {len(id2label)} classi disponibili")

# Mostra le classi
print("\n📋 Classi del modello:")
unique_types = set()
for label in id2label.values():
    entity_type = label.split("-")[-1] if "-" in label else label
    if entity_type != "O":
        unique_types.add(entity_type)
print(f"   {', '.join(sorted(unique_types))}")


# ============================================
# FUNZIONI CORE
# ============================================


def predict_and_redact(text: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """Predice entità e restituisce testo censurato."""

    # Tokenizza
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Predici
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values

    # Estrai entità con logica più permissiva
    entities = []
    current_entity = None

    for pred_id, confidence, offset in zip(predictions, confidences, offset_mapping):
        if offset[0] == offset[1]:
            continue

        label = id2label[pred_id.item()]
        conf = confidence.item()

        if label != "O":
            entity_type = label.split("-")[-1] if "-" in label else label

            # Logica più aggressiva per continuare entità
            if (
                label.startswith("B-")
                or not current_entity
                or entity_type != current_entity["type"]
            ):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "type": entity_type,
                    "start": offset[0].item(),
                    "end": offset[1].item(),
                    "confidences": [conf],
                }
            else:
                # Continua entità
                if current_entity:
                    current_entity["end"] = offset[1].item()
                    current_entity["confidences"].append(conf)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Calcola confidence media e filtra
    for entity in entities:
        entity["confidence"] = np.mean(entity["confidences"])
    entities = [e for e in entities if e["confidence"] >= confidence_threshold]

    # Censura (dall'ultima alla prima)
    redacted = text
    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
        start, end = entity["start"], entity["end"]
        replacement = REDACTION_CHAR * (end - start)
        redacted = redacted[:start] + replacement + redacted[end:]

    return {
        "original": text,
        "redacted": redacted,
        "entities": [
            (text[e["start"] : e["end"]], e["type"], e["confidence"]) for e in entities
        ],
    }


def test(text: str, title: str = "", threshold: float = CONFIDENCE_THRESHOLD):
    """Test rapido con output essenziale."""
    print(f"\n{'=' * 70}")
    if title:
        print(f"📝 {title}")
        print(f"{'=' * 70}")

    result = predict_and_redact(text, threshold)

    print(
        f"\n🔓 Originale:\n{result['original'][:200]}{'...' if len(result['original']) > 200 else ''}"
    )

    if result["entities"]:
        print(f"\n🔍 Entità trovate ({len(result['entities'])}):")
        for entity_text, entity_type, conf in result["entities"][:10]:  # Max 10
            display_text = (
                entity_text[:30] + "..." if len(entity_text) > 30 else entity_text
            )
            print(f"  • {display_text:<33} [{entity_type:<15}] {conf:.1%}")
        if len(result["entities"]) > 10:
            print(f"  ... e altre {len(result['entities']) - 10} entità")
    else:
        print("\n⚠️  Nessuna entità trovata sopra la soglia")

    print(
        f"\n🔒 Censurato:\n{result['redacted'][:200]}{'...' if len(result['redacted']) > 200 else ''}"
    )


# ============================================
# ESEMPI REALISTICI (stile dataset finanziario)
# ============================================

# Esempio 1: Contratto di prestito
test(
    """
CONTRATTO DI PRESTITO PERSONALE

Tra le parti:
Mutuatario: Mario Rossi, nato il 15/03/1985, residente in Via Roma 123, Milano
Mutuante: Banca Intesa S.p.A., sede legale in Piazza Cordusio 1, Milano

Importo: EUR 50.000,00
Tasso di interesse: 5,5% annuo
Durata: 60 mesi
Numero contratto: IT-2024-001234

Il Mutuatario si impegna a rimborsare l'importo secondo il piano di ammortamento allegato.
Primo pagamento: 01/04/2024
IBAN: IT60X0542811101000000123456

Firma del Mutuatario: _________________
Data: 15/03/2024
""".strip(),
    "Contratto di Prestito",
)

# Esempio 2: Polizza assicurativa
test(
    """
POLIZZA ASSICURATIVA AUTO N. POL-2024-5678

Contraente: Giulia Bianchi
Data di nascita: 10/08/1990
Indirizzo: Corso Vittorio Emanuele 45, Torino 10121
Codice Fiscale: BNCGLI90M50L219X
Telefono: +39 011 5551234
Email: giulia.bianchi@email.it

Veicolo assicurato:
Targa: AB123CD
Modello: Fiat 500

Premio annuo: EUR 850,00
Scadenza: 31/12/2024
Compagnia: UnipolSai Assicurazioni S.p.A.
""".strip(),
    "Polizza Assicurativa",
)

# Esempio 3: Estratto conto
test(
    """
ESTRATTO CONTO BANCARIO

Intestatario: Luca Verdi
IBAN: IT28W8000000292100645211
Periodo: 01/01/2024 - 31/01/2024

Saldo iniziale: EUR 5.234,56
Movimenti:
- 05/01 Bonifico da Anna Neri - EUR 1.200,00
- 12/01 Pagamento ENEL Energia - EUR -85,40
- 20/01 Prelievo Bancomat Via Garibaldi 12 - EUR -100,00
- 25/01 Stipendio Tech Solutions S.r.l. - EUR 2.500,00

Saldo finale: EUR 8.749,16

Per informazioni: clienti@bancaintesa.it
Tel: 800-123456
""".strip(),
    "Estratto Conto",
)

# Esempio 4: Richiesta di finanziamento
test(
    """
DOMANDA DI FINANZIAMENTO

Richiedente: Francesco Colombo
Luogo e data di nascita: Roma, 22/11/1982
Documento: Carta d'Identità n. CA1234567 rilasciata da Comune di Roma
Residenza: Via Nazionale 200, 00184 Roma
Partita IVA: 12345678901
Telefono cellulare: +39 333 9876543

Importo richiesto: EUR 75.000,00
Finalità: Acquisto macchinari aziendali
Garanzie offerte: Ipoteca su immobile commerciale in Via Veneto 89, Roma

Dati azienda:
Ragione sociale: Colombo Costruzioni S.r.l.
Sede: Viale Mazzini 150, Roma
""".strip(),
    "Domanda di Finanziamento",
)

# Esempio 5: Fattura
test(
    """
FATTURA N. 2024/0042

Spett.le
Roberto Gentile
Via Manzoni 78
20121 Milano
P.IVA: 09876543210

Data fattura: 10/03/2024
Pagamento: Bonifico bancario
IBAN: IT45L0300203280284975662883

Descrizione servizi:
- Consulenza legale (15 ore) - EUR 2.250,00
- Spese vive e oneri accessori - EUR 150,00

Totale imponibile: EUR 2.400,00
IVA 22%: EUR 528,00
Totale fattura: EUR 2.928,00

Studio Legale Associato Rossi & Partners
Via della Repubblica 45, Milano
Tel: 02-12345678 | Email: info@studiorossi.it
""".strip(),
    "Fattura Professionale",
)


# ============================================
# TEST CON THRESHOLD DIVERSI
# ============================================

print(f"\n{'=' * 70}")
print("🎯 ANALISI THRESHOLD")
print(f"{'=' * 70}")

sample_text = "Il Sig. Mario Rossi (CF: RSSMRA85C15F205X) residente in Via Roma 15, Milano, ha sottoscritto un contratto con Banca Intesa."

for thresh in [0.2, 0.3, 0.4, 0.5]:
    result = predict_and_redact(sample_text, thresh)
    print(f"\nThreshold {thresh:.1f} → {len(result['entities'])} entità")
    if result["entities"]:
        for text, etype, conf in result["entities"]:
            print(f"  • {text} [{etype}] ({conf:.1%})")


# ============================================
# MODALITÀ INTERATTIVA
# ============================================


# def interactive():
#     """Modalità interattiva."""
#     print(f"\n{'=' * 70}")
#     print("🎮 MODALITÀ INTERATTIVA")
#     print(f"{'=' * 70}")
#     print("Inserisci un documento finanziario (lascia vuoto per uscire)\n")

#     while True:
#         try:
#             text = input("\n📝 Testo (o 'quit'): ").strip()
#             if not text or text.lower() == "quit":
#                 break

#             thresh_input = input(
#                 f"⚙️  Threshold (default {CONFIDENCE_THRESHOLD}): "
#             ).strip()
#             threshold = float(thresh_input) if thresh_input else CONFIDENCE_THRESHOLD

#             result = predict_and_redact(text, threshold)

#             print(f"\n🔍 {len(result['entities'])} entità trovate")
#             for ent_text, ent_type, conf in result["entities"]:
#                 print(f"  • {ent_text[:40]} [{ent_type}] ({conf:.1%})")

#             print(f"\n🔒 Censurato:\n{result['redacted']}")

#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             print(f"⚠️  Errore: {e}")

#     print("\n👋 Fine")


# Chiedi modalità interattiva
# print(f"\n{'=' * 70}")
# response = input("\n▶️  Vuoi testare documenti personalizzati? (y/n): ").strip().lower()
# if response == "y":
#     interactive()
# else:
#     print("\n✅ Test completati!")
