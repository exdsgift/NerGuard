from datasets import load_from_disk
from transformers import AutoTokenizer

class Colors:
   OKGREEN = '\033[92m'
   FAIL = '\033[91m'
   WARNING = '\033[93m'
   ENDC = '\033[0m'
   BOLD = '\033[1m'

def check_data_alignment():
   DATA_PATH = "./data/processed/tokenized_data"
   dataset = load_from_disk(DATA_PATH)
   
   # Carica il tokenizer che hai usato
   tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
   
   # Prendi il primo esempio che ha delle entità (non solo O)
   print("--- ISPEZIONE VISIVA DATASET ---")
   found = 0
   for i in range(len(dataset["train"])):
      row = dataset["train"][i]
      labels = row["labels"]
      
      # Se c'è almeno una label diversa da 0 (O) e -100 (IGN)
      if any(l > 0 for l in labels):
         tokens = tokenizer.convert_ids_to_tokens(row["input_ids"])
         print(f"\nESEMPIO {i} (Trovato nel Train Set):")
         
         for tok, lbl in zip(tokens, labels):
               if lbl == -100: lbl_str = "IGN"
               elif lbl == 0: lbl_str = "O"
               else: lbl_str = f"ENTITY-{lbl}" # Non ci serve il nome preciso, basta vedere se non è O
               
               # Stampa solo se è un'entità o vicino a un'entità
               if lbl_str != "O" or "ENTITY" in lbl_str:
                  print(f"{tok:<15} | {lbl_str}")
         
         found += 1
         if found >= 3: break # Guardiamo solo 3 esempi

def debug_decode_entities(dataset, tokenizer, num_samples=10):
   print(f"\n{Colors.BOLD}--- DECODING CHECK (Cosa vede il modello?) ---{Colors.ENDC}")
   count = 0
   for i in range(len(dataset["train"])):
      labels = dataset["train"][i]["labels"]
      if not any(l > 0 for l in labels): continue # Salta frasi vuote
      
      ids = dataset["train"][i]["input_ids"]
      tokens = tokenizer.convert_ids_to_tokens(ids)
      
      print(f"\nEsempio {i}:")
      # Ricostruiamo visivamente
      reconstructed = []
      for token, label in zip(tokens, labels):
         if label > 0: # È un'entità
               reconstructed.append(f"{Colors.OKGREEN}{token}{Colors.ENDC}")
         elif label == -100:
               pass
         else:
               reconstructed.append(token)
      
      print(" ".join(reconstructed).replace(" ##", ""))
      
      count += 1
      if count >= num_samples: break
      
if __name__ == "__main__":
   DATA_PATH = "./data/processed/tokenized_data"
   dataset = load_from_disk(DATA_PATH)
   
   # Carica il tokenizer che hai usato
   tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
   # check_data_alignment()
   debug_decode_entities(dataset, tokenizer)