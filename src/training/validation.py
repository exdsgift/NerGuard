import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import os
import sys
from dotenv import load_dotenv


current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
   sys.path.insert(0, project_root)

load_dotenv()

from src.pipeline.entropy_inference import LLMRouter  # noqa: E402

MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
OUTPUT_DIR = "./evaluation_results"

THRESHOLD_ENTROPY = 0.3 
THRESHOLD_CONF = 0.85

SAMPLE_LIMIT = 500

def ensure_dir(path):
   if not os.path.exists(path):
      os.makedirs(path)

def plot_confusion_matrix(y_true, y_pred, id2label, title, filename):
   unique_labels = sorted(list(set(y_true) | set(y_pred)))
   label_names = [id2label[i] for i in unique_labels]

   cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
   with np.errstate(divide='ignore', invalid='ignore'):
      cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   cm_norm = np.nan_to_num(cm_norm)

   plt.figure(figsize=(14, 12))
   sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.title(title)
   plt.tight_layout()
   plt.savefig(filename)
   plt.close()
   print(f"   [PLOT] Saved: {filename}")

def evaluate():
   print("HYBRID EVAL")
   print(f"Model: {MODEL_PATH}")
   ensure_dir(OUTPUT_DIR)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
   model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
   model.eval()
   
   router = LLMRouter(source="openai") 
   
   print("   Loading Validation Dataset...")
   dataset = load_from_disk(DATA_PATH)
   eval_dataset = dataset["validation"]
   
   if SAMPLE_LIMIT:
      print(f"   ⚠️ TEST MODE: Limiting to first {SAMPLE_LIMIT} samples.")
      eval_dataset = eval_dataset.select(range(SAMPLE_LIMIT))

   id2label = model.config.id2label
   label2id = model.config.label2id

   # Buffers
   baseline_preds = []
   hybrid_preds = []
   true_labels = []
   
   llm_intervention_count = 0
   llm_correction_count = 0

   print("\nRunning Inference Loop")
   
   for i in tqdm(range(len(eval_dataset)), desc="Processing"):
      sample = eval_dataset[i]
      
      input_ids = torch.tensor([sample["input_ids"]]).to(device)
      attention_mask = torch.tensor([sample["attention_mask"]]).to(device)
      labels = sample["labels"]

      full_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
   
      tokenized_ref = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
      offsets = tokenized_ref["offset_mapping"]
      tokens_str = tokenizer.convert_ids_to_tokens(sample["input_ids"])

      with torch.no_grad():
         outputs = model(input_ids, attention_mask=attention_mask)
      
      logits = outputs.logits[0]
      probs = F.softmax(logits, dim=-1)
      log_probs = F.log_softmax(logits, dim=-1)
      
      entropy = -torch.sum(probs * log_probs, dim=-1)
      confidences, pred_ids = torch.max(probs, dim=-1)
      
      pred_ids_list = pred_ids.cpu().tolist()
      conf_list = confidences.cpu().tolist()
      entr_list = entropy.cpu().tolist()
      
      prev_label_hybrid = "O"
      seq_len = len(labels)
      
      offset_idx = 0 

      for j in range(seq_len):
         lbl = labels[j]
         token_str = tokens_str[j]
         
         if lbl == -100: 
               continue
         
         if token_str in ['[CLS]', '[SEP]', '[PAD]']:
               continue

         base_pred_id = pred_ids_list[j]
         baseline_preds.append(base_pred_id)
         true_labels.append(lbl)

         hybrid_pred_id = base_pred_id
         curr_base_label = id2label[base_pred_id]
         
         conf = conf_list[j]
         ent = entr_list[j]

         char_start, char_end = 0, 0
         if offset_idx < len(offsets):
               char_start, char_end = offsets[offset_idx]
         offset_idx += 1

         if ent > THRESHOLD_ENTROPY and conf < THRESHOLD_CONF:
               llm_intervention_count += 1
               try:
                  llm_result = router.disambiguate(
                     target_token=token_str,
                     full_text=full_text,
                     char_start=char_start,
                     char_end=char_end,
                     current_pred=curr_base_label,
                     prev_label=prev_label_hybrid,
                     lang="en"
                  )
                  
                  if llm_result.get("is_pii"):
                     corrected_label = llm_result.get("corrected_label", "O")
                     if corrected_label in label2id:
                           hybrid_pred_id = label2id[corrected_label]
                           if hybrid_pred_id != base_pred_id:
                              llm_correction_count += 1
                     
               except Exception as e:
                  pass
         
         hybrid_preds.append(hybrid_pred_id)
         prev_label_hybrid = id2label[hybrid_pred_id]

   print("\n" + "="*60)
   print("EVAL REPORT")
   print("="*60)
   print(f"Total Tokens Evaluated: {len(true_labels)}")
   print(f"LLM Calls Triggered: {llm_intervention_count}")
   print(f"LLM Actual Corrections (Changed Label): {llm_correction_count}")
   print("="*60)

   active_labels = sorted(list(set(true_labels)))
   target_names = [id2label[i] for i in active_labels]

   print("\nDeBERTa v3")
   print(classification_report(true_labels, baseline_preds, labels=active_labels, target_names=target_names, digits=4))

   print("\nDeBERTa + LLM")
   print(classification_report(true_labels, hybrid_preds, labels=active_labels, target_names=target_names, digits=4))

   # Plot
   print("\nGenerating Confusion Matrices")
   plot_confusion_matrix(true_labels, baseline_preds, id2label, "Baseline (Standard)", f"{OUTPUT_DIR}/cm_baseline.png")
   plot_confusion_matrix(true_labels, hybrid_preds, id2label, "Hybrid (Ours)", f"{OUTPUT_DIR}/cm_hybrid.png")
   
   # Sace report
   with open(f"{OUTPUT_DIR}/report.txt", "w") as f:
      f.write("BASELINE\n")
      f.write(classification_report(true_labels, baseline_preds, labels=active_labels, target_names=target_names, digits=4))
      f.write("\n\nHYBRID\n")
      f.write(classification_report(true_labels, hybrid_preds, labels=active_labels, target_names=target_names, digits=4))

if __name__ == "__main__":
   evaluate()