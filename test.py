from transformers import (AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, DataCollatorWithPadding,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          TokenClassificationPipeline)
from data_processing import upload_dataset
import numpy as np
import pprint as pp
import os
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import accuracy_score, f1_score
import pprint
import json


SAVE_DIR = "pii_mdeberta_softmax/outputs/best"
DATASET_PATH = "data/processed_dataset"
BATCH_SIZE = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# * Smoke test to check if the model/checkpoint is working
# tok = AutoTokenizer.from_pretrained(SAVE_DIR)
# mdl = AutoModelForTokenClassification.from_pretrained(SAVE_DIR)

# pipe = TokenClassificationPipeline(model=mdl, tokenizer=tok, aggregation_strategy="simple", device=3)
# pp.pprint(pipe("John Doe lives at 221B Baker Street, London and his email is john@example.com"))


# * Trainer Evaluation

print(f"📂 Caricamento modello da: {SAVE_DIR}")
print(f"📂 Caricamento dataset da: {DATASET_PATH}")

# Carica modello e tokenizer
config = AutoConfig.from_pretrained(SAVE_DIR)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

# Carica dataset
ds = upload_dataset(DATASET_PATH)

# Determina il tipo di task
is_token_classification = config.architectures and any(
    "TokenClassification" in a for a in config.architectures
)

print(f"🔍 Task type: {'Token Classification' if is_token_classification else 'Sequence Classification'}")

# Configura modello e metrics
if is_token_classification:
    model = AutoModelForTokenClassification.from_pretrained(SAVE_DIR)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Import corretto per seqeval
    from seqeval.metrics import precision_score, recall_score, f1_score as seqeval_f1, accuracy_score as seqeval_accuracy
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Rimuovi padding (-100)
        true_labels = [[label for label in seq if label != -100] for seq in labels]
        true_preds = [[pred for pred, label in zip(seq_pred, seq_label) if label != -100]
                      for seq_pred, seq_label in zip(predictions, labels)]
        
        # Converti in stringhe se hai id2label
        if hasattr(config, 'id2label') and config.id2label:
            true_labels = [[config.id2label[l] for l in seq] for seq in true_labels]
            true_preds = [[config.id2label[p] for p in seq] for seq in true_preds]
        
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": seqeval_f1(true_labels, true_preds),
            "accuracy": seqeval_accuracy(true_labels, true_preds),
        }
else:
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

# Configura Trainer
args = TrainingArguments(
    output_dir="tmp_eval",
    per_device_eval_batch_size=BATCH_SIZE,
    dataloader_num_workers=4,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    processing_class=tokenizer,  # Usa processing_class invece di tokenizer
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Funzione helper per stampare risultati
def print_metrics(split_name, metrics):
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} METRICS")
    print('='*60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {key:30s}: {value:8.4f}")
        else:
            print(f"  {key:30s}: {value}")
    print()

# Valuta su tutti i split
results = {}

# Training set
print("\n🔄 Evaluating on TRAIN set...")
train_metrics = trainer.evaluate(eval_dataset=ds['train'])
results['train'] = train_metrics
print_metrics("Training", train_metrics)

# Validation set
print("🔄 Evaluating on VALIDATION set...")
val_metrics = trainer.evaluate(eval_dataset=ds['validation'])
results['validation'] = val_metrics
print_metrics("Validation", val_metrics)

# Test set
if 'test' in ds:
    print("🔄 Evaluating on TEST set...")
    test_metrics = trainer.evaluate(eval_dataset=ds['test'])
    results['test'] = test_metrics
    print_metrics("Test", test_metrics)
    
    # Predizioni complete
    print("🔄 Getting predictions on TEST set...")
    predictions = trainer.predict(ds['test'])
    print(f"  Prediction shape: {predictions.predictions.shape}")
    print(f"  Labels shape: {predictions.label_ids.shape}")

# Riepilogo comparativo
print("\n" + "="*60)
print("  COMPARATIVE SUMMARY")
print("="*60)

# Trova metriche disponibili
key_metrics = ['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_f1_macro', 
               'eval_f1_weighted', 'eval_precision', 'eval_recall']
available_metrics = set()
for split_metrics in results.values():
    available_metrics.update(split_metrics.keys())
key_metrics = [m for m in key_metrics if m in available_metrics]

# Stampa tabella comparativa
if key_metrics:
    # Header
    print(f"\n{'Metric':<20}", end="")
    splits_available = [s for s in ['train', 'validation', 'test'] if s in results]
    for split in splits_available:
        print(f"{split:>15}", end="")
    print()
    print("-" * (20 + 15 * len(splits_available)))
    
    # Righe
    for metric in key_metrics:
        metric_name = metric.replace('eval_', '')
        print(f"{metric_name:<20}", end="")
        for split in splits_available:
            if metric in results[split]:
                print(f"{results[split][metric]:15.4f}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()

# Calcola overfitting indicator
if 'train' in results and 'validation' in results:
    print("\n" + "="*60)
    print("  OVERFITTING ANALYSIS")
    print("="*60)
    
    for metric in ['eval_accuracy', 'eval_f1', 'eval_f1_macro']:
        if metric in results['train'] and metric in results['validation']:
            train_val = results['train'][metric]
            val_val = results['validation'][metric]
            gap = train_val - val_val
            gap_pct = (gap / train_val * 100) if train_val > 0 else 0
            
            status = "✅ Good" if abs(gap_pct) < 5 else "⚠️  Warning" if abs(gap_pct) < 15 else "🔴 Overfitting"
            print(f"{metric.replace('eval_', ''):20s}: Train={train_val:.4f}, Val={val_val:.4f}, Gap={gap:+.4f} ({gap_pct:+.1f}%) {status}")

# Salva risultati
output_file = 'results/evaluation_results.json'
results_to_save = {
    split: {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
            for k, v in metrics.items() if not k.startswith('eval_runtime')}
    for split, metrics in results.items()
}

with open(output_file, 'w') as f:
    json.dump(results_to_save, f, indent=2)

print(f"\n✅ Results saved to '{output_file}'")
print(f"✅ Evaluation complete!")

# Opzionale: crea grafici
try:
    import matplotlib.pyplot as plt
    
    print("\n📊 Creating visualization...")
    
    # Prepara dati per grafici
    splits_available = [s for s in ['train', 'validation', 'test'] if s in results]
    
    # Trova metriche da plottare
    metrics_to_plot = []
    for m in ['eval_accuracy', 'eval_f1', 'eval_f1_macro', 'eval_f1_weighted', 
              'eval_precision', 'eval_recall']:
        if any(m in results[s] for s in splits_available):
            metrics_to_plot.append(m)
    
    if metrics_to_plot:
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        colors = {'train': '#3498db', 'validation': '#e74c3c', 'test': '#2ecc71'}
        
        for idx, metric in enumerate(metrics_to_plot):
            values = []
            labels = []
            colors_list = []
            
            for split in splits_available:
                if metric in results[split]:
                    values.append(results[split][metric])
                    labels.append(split.capitalize())
                    colors_list.append(colors[split])
            
            axes[idx].bar(labels, values, color=colors_list, alpha=0.8, edgecolor='black')
            axes[idx].set_title(metric.replace('eval_', '').replace('_', ' ').title(), 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylim(0, 1)
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Aggiungi valori sopra le barre
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/model_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to 'model_metrics_comparison.png'")
        plt.close()
    
except ImportError:
    print("\n⚠️  matplotlib not installed. Skipping visualization.")
    print("   Install with: uv pip install matplotlib")