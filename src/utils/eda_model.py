import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import os
import sys

# --- CONFIGURAZIONE ---
MODEL_PATH = "./models/mdeberta-pii-safe/final"
LABEL_PATH = "./data/processed/id2label.json" 
THRESHOLD = 0.0303  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PIITester:
   def __init__(self, model_path, label_path):
      self.device = DEVICE
      print(f"🌍 Inizializzazione Sistema Multilingua su {self.device}...")
      self.id2label = self._load_labels(label_path)
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)
      self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
      self.model.eval()

   def _load_labels(self, path):
      if not os.path.exists(path):
         sys.exit(f"File label not found: {path}")
      with open(path, "r") as f:
         return {int(k): v for k, v in json.load(f).items()}

   def analyze_sentence(self, lang_code, text):
      inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
      
      with torch.no_grad():
         logits = self.model(**inputs).logits[0]

      probs = F.softmax(logits, dim=-1)
      log_probs = F.log_softmax(logits, dim=-1)
      entropy = -torch.sum(probs * log_probs, dim=-1)
      
      confidences, pred_ids = torch.max(probs, dim=-1)
      tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
      
      # Report
      print(f"\n[{lang_code.upper()}] Frase: '{text}'\n")
      
      max_ent = 0.0
      has_pii = False
      
      print(f"   {'TOKEN':<25}\t{'PRED':<20}\t{'CONF':<8}\t{'ENTR':<8}\t{'STATUS'}")
      print("   " + "_" * 100)

      for token, pid, conf, ent in zip(tokens, pred_ids, confidences, entropy):
         if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
         
         label = self.id2label.get(pid.item(), "UNK")
         clean_token = token.replace(' ', '')
         
         # Tracking
         if ent.item() > max_ent: max_ent = ent.item()
         is_uncertain = ent.item() > THRESHOLD
         
         # Mostra solo se è PII o se è Incerto (nasconde i token "O" sicuri per pulizia)
         if label != 'O' or is_uncertain:
               flag = "⚠️" if is_uncertain else "✅"
               print(f"   {clean_token:<25}\t{label:<20}\t{conf.item():.2f}\t\t{ent.item():.3f}\t\t{flag}")
      
      if max_ent > THRESHOLD:
         print(f"\n   --> ROUTING ATTIVATO (Max Ent: {max_ent:.4f}) -> LLM richiamato per contesto {lang_code}.\n\n")
      else:
         print(f"\n   --> GESTIONE LOCALE (Max Ent: {max_ent:.4f}) -> Encoder sicuro.\n\n")

if __name__ == "__main__":
   tester = PIITester(MODEL_PATH, LABEL_PATH)
   
   multilingual_cases = [
      # English (120k samples - High Resource)
      ("en", "Please send the report to Mr. John Smith at j.smith@company.com immediately."),
      
      # French (89k samples - High Resource)
      ("fr", "J'habite au 15 Rue de la Paix, Paris. Mon nom est Pierre Martin."),
      
      # German (65k samples - Medium Resource)
      ("de", "Mein Name ist Thomas Müller und ich lebe in der Berliner Straße 5, München."),
      
      # Spanish (62k samples - Medium Resource)
      ("es", "La doctora Ana María González López trabaja en el Hospital Central de Madrid."),
      
      # Italian (55k samples - Medium Resource)
      ("it", "Il codice fiscale di Mario Rossi è RSSMRA80A01H501U."),
      
      # Hindi (27k samples - Low Resource / Different Script)
      ("hi", "मेरा नाम राहुल कुमार है और मैं दिल्ली में रहता हूँ।"),
      
      # Telugu (22k samples - Low Resource / Agglutinative)
      ("te", "నా పేరు రవి, నా ఫోన్ నంబర్ 9848022338."),
      
      # Dutch (21k samples - Low Resource)
      ("nl", "Ik ben Sven van der Berg en mijn e-mailadres is sven.berg@example.nl.")
   ]
   
   print(f"\n🚀 AVVIO TEST MULTILINGUA ({len(multilingual_cases)} lingue)\n")
   
   for lang, text in multilingual_cases:
      tester.analyze_sentence(lang, text)