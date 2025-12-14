import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

# Configurazione
MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
BATCH_SIZE = 32

def plot_confusion_matrix(y_true, y_pred, labels, output_file="confusion_matrix.png"):
   # Rimuovi la label 'O' per chiarezza se domina troppo (opzionale, qui la teniamo ma occhio ai numeri)
   # Filtra solo le label che appaiono effettivamente
   unique_labels = sorted(list(set(y_true) | set(y_pred)))
   label_names = [labels[i] for i in unique_labels]

   cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
   
   # Normalizza per righe (Recall per classe)
   cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   
   plt.figure(figsize=(12, 10))
   sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.title('Normalized Confusion Matrix')
   plt.tight_layout()
   plt.savefig(output_file)
   print(f"Matrice salvata in {output_file}")
   plt.show()

def evaluate_model():
   print("Loading resources...")
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Load Model & Tokenizer
   try:
      tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
      model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
      model.to(device)
      model.eval()
   except Exception as e:
      print(f"Error loading model: {e}")
      return

   # Load Validation Set
   dataset = load_from_disk(DATA_PATH)
   eval_dataset = dataset["validation"] # Usa 'test' se esiste, altrimenti 'validation'
   
   id2label = model.config.id2label
   
   all_preds = []
   all_labels = []
   
   print("Running Inference on Validation Set...")
   # Inference Loop (non usiamo Trainer per avere controllo raw)
   for i in tqdm(range(0, len(eval_dataset), BATCH_SIZE)):
      batch = eval_dataset[i : i + BATCH_SIZE]
      
      # Pad batch manuale o via tokenizer
      inputs = tokenizer.pad(
         {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
         return_tensors="pt",
         padding=True
      ).to(device)
      
      with torch.no_grad():
         outputs = model(**inputs)
      
      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      
      # Srotoliamo i batch
      for j in range(len(batch["input_ids"])):
         pred_seq = predictions[j].cpu().numpy()
         label_seq = batch["labels"][j] # label_seq è una lista di int
         
         # Filtra i pad tokens e gli ignored index (-100)
         valid_indices = [idx for idx, l in enumerate(label_seq) if l != -100]
         
         filtered_preds = pred_seq[valid_indices]
         filtered_labels = [label_seq[idx] for idx in valid_indices]
         
         all_preds.extend(filtered_preds)
         all_labels.extend(filtered_labels)

   print("\n--- Generating Report ---")
   
   # Classification Report Testuale
   label_ids = sorted(list(set(all_labels)))
   target_names = [id2label[i] for i in label_ids]
   
   print(classification_report(all_labels, all_preds, labels=label_ids, target_names=target_names, digits=4))
   
   # Confusion Matrix Plot
   plot_confusion_matrix(all_labels, all_preds, id2label)

if __name__ == "__main__":
   evaluate_model()