import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

# --- CONFIGURAZIONE RIPRODUCIBILE ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
OUTPUT_DIR = "./optimization_plots"
BATCH_SIZE = 32
SAMPLE_LIMIT = None

# Parametri economici (da citare nelle assunzioni della tesi)
LLM_COST_PER_TOKEN = 0.00002  # GPT-4 Turbo pricing (~$0.02/1K tokens)
ERROR_COST_PER_TOKEN = 0.001   # Costo di violazione privacy/GDPR (stima conservativa)

# F-beta: privilegia precision (riduce falsi positivi = sprechi)
BETA_SCORE = 0.5

# Grid search
GRID_RESOLUTION = 30

# Bootstrap per confidence intervals
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

@dataclass
class OptimizationResult:
   """Risultati con statistiche complete per tesi."""
   ent_threshold: float
   conf_threshold: float
   precision: float
   recall: float
   f_score: float
   intervention_rate: float
   expected_cost: float
   cost_savings: float
   # Confidence intervals (2.5%, 97.5% per CI 95%)
   precision_ci: Tuple[float, float] = None
   recall_ci: Tuple[float, float] = None
   f_score_ci: Tuple[float, float] = None
   # Dettagli per tabella LaTeX
   n_samples: int = 0
   n_errors: int = 0
   n_triggers: int = 0
   true_positives: int = 0
   false_positives: int = 0
   false_negatives: int = 0

def ensure_dir(path):
   os.makedirs(path, exist_ok=True)

def set_style():
   """Stile professionale per pubblicazioni."""
   sns.set_theme(style="whitegrid", context="paper")
   plt.rcParams.update({
      "font.family": "serif",
      # Fallback fonts disponibili su Linux/Mac/Windows
      "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
      "figure.dpi": 300,
      "savefig.dpi": 300,
      "axes.titlesize": 14,
      "axes.labelsize": 12,
      "xtick.labelsize": 10,
      "ytick.labelsize": 10,
      "legend.fontsize": 10,
      "figure.titlesize": 16
   })

def collect_inference_stats_batched(model, dataset, device):
   """
   Inferenza con logging dettagliato per reproducibilità.
   """
   all_entropies = []
   all_confidences = []
   all_is_error = []
   
   print("   Running Batched Inference...")
   
   for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Inference"):
      batch = dataset[i:i + BATCH_SIZE]
      
      max_len = max(len(x) for x in batch["input_ids"])
      
      input_ids = torch.tensor([
         x + [0] * (max_len - len(x)) for x in batch["input_ids"]
      ]).to(device)
      
      attention_mask = torch.tensor([
         x + [0] * (max_len - len(x)) for x in batch["attention_mask"]
      ]).to(device)
      
      labels = torch.tensor([
         x + [-100] * (max_len - len(x)) for x in batch["labels"]
      ]).to(device)
      
      with torch.no_grad():
         outputs = model(input_ids, attention_mask=attention_mask)
      
      probs = F.softmax(outputs.logits, dim=-1)
      log_probs = F.log_softmax(outputs.logits, dim=-1)
      
      entropy = -torch.sum(probs * log_probs, dim=-1)
      confidences, pred_ids = torch.max(probs, dim=-1)
      
      mask = labels != -100
      is_error = (pred_ids != labels) & mask
      
      for b in range(len(batch["input_ids"])):
         valid_mask = mask[b]
         if valid_mask.sum() == 0:
               continue
         
         all_entropies.extend(entropy[b][valid_mask].cpu().numpy())
         all_confidences.extend(confidences[b][valid_mask].cpu().numpy())
         all_is_error.extend(is_error[b][valid_mask].cpu().numpy())
   
   return np.array(all_entropies), np.array(all_confidences), np.array(all_is_error)

def bootstrap_metric(trigger_mask, is_errors, metric_fn, n_bootstrap=N_BOOTSTRAP):
   """
   Calcola confidence interval via bootstrap per qualsiasi metrica.
   Essenziale per validità statistica nella tesi.
   """
   metrics = []
   n_samples = len(is_errors)
   
   for _ in range(n_bootstrap):
      # Resample con sostituzione
      indices = np.random.choice(n_samples, size=n_samples, replace=True)
      sampled_triggers = trigger_mask[indices]
      sampled_errors = is_errors[indices]
      
      metric_value = metric_fn(sampled_triggers, sampled_errors)
      metrics.append(metric_value)
   
   metrics = np.array(metrics)
   alpha = 1 - CONFIDENCE_LEVEL
   ci_low = np.percentile(metrics, 100 * alpha / 2)
   ci_high = np.percentile(metrics, 100 * (1 - alpha / 2))
   
   return ci_low, ci_high

def calculate_expected_cost(intervention_rate, recall, n_errors, n_total):
   """Costo atteso: LLM calls + missed errors."""
   n_llm_calls = intervention_rate * n_total
   n_missed_errors = (1 - recall) * n_errors
   
   llm_cost = n_llm_calls * LLM_COST_PER_TOKEN
   error_cost = n_missed_errors * ERROR_COST_PER_TOKEN
   
   return llm_cost + error_cost

def vectorized_grid_search(entropies, confidences, is_errors):
   """Grid search completa con tutte le metriche per analisi."""
   print("\n   Calculating Optimization Matrix...")
   
   ent_grid = np.linspace(0.1, 1.5, GRID_RESOLUTION)
   conf_grid = np.linspace(0.5, 0.99, GRID_RESOLUTION)
   
   results = {
      "f_score": np.zeros((len(ent_grid), len(conf_grid))),
      "precision": np.zeros((len(ent_grid), len(conf_grid))),
      "recall": np.zeros((len(ent_grid), len(conf_grid))),
      "intervention_rate": np.zeros((len(ent_grid), len(conf_grid))),
      "expected_cost": np.zeros((len(ent_grid), len(conf_grid))),
      "specificity": np.zeros((len(ent_grid), len(conf_grid)))  # Per ROC
   }
   
   total_samples = len(is_errors)
   total_errors = np.sum(is_errors)
   total_correct = total_samples - total_errors
   
   baseline_cost = total_samples * LLM_COST_PER_TOKEN
   
   if total_errors == 0:
      raise ValueError("Dataset non contiene errori! Verifica i dati.")
   
   for i, ent_th in enumerate(tqdm(ent_grid, desc="Grid Search")):
      for j, conf_th in enumerate(conf_grid):
         trigger_mask = (entropies > ent_th) & (confidences < conf_th)
         n_triggers = np.sum(trigger_mask)
         
         if n_triggers == 0:
               results["precision"][i, j] = 0
               results["recall"][i, j] = 0
               results["f_score"][i, j] = 0
               results["intervention_rate"][i, j] = 0
               results["expected_cost"][i, j] = total_errors * ERROR_COST_PER_TOKEN
               results["specificity"][i, j] = 1.0  # No false positives
               continue
         
         tp = np.sum(trigger_mask & is_errors)
         fp = np.sum(trigger_mask & ~is_errors)
         fn = np.sum(~trigger_mask & is_errors)
         tn = np.sum(~trigger_mask & ~is_errors)
         
         precision = tp / n_triggers if n_triggers > 0 else 0
         recall = tp / total_errors
         specificity = tn / total_correct if total_correct > 0 else 0
         
         # F-beta
         if precision + recall > 0:
               beta_sq = BETA_SCORE ** 2
               f_score = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
         else:
               f_score = 0
         
         intervention_rate = n_triggers / total_samples
         expected_cost = calculate_expected_cost(intervention_rate, recall, total_errors, total_samples)
         
         results["f_score"][i, j] = f_score
         results["precision"][i, j] = precision
         results["recall"][i, j] = recall
         results["intervention_rate"][i, j] = intervention_rate
         results["expected_cost"][i, j] = expected_cost
         results["specificity"][i, j] = specificity
   
   results["baseline_cost"] = baseline_cost
   return ent_grid, conf_grid, results

def plot_heatmap(ent_grid, conf_grid, matrix, best_idx, metric_name, filename):
   """Heatmap publication-quality."""
   plt.figure(figsize=(10, 8))
   
   X, Y = np.meshgrid(conf_grid, ent_grid)
   c = plt.pcolormesh(X, Y, matrix, shading='auto', cmap='viridis')
   cbar = plt.colorbar(c, label=metric_name)
   
   best_i, best_j = best_idx
   best_ent = ent_grid[best_i]
   best_conf = conf_grid[best_j]
   plt.scatter(best_conf, best_ent, color='red', s=150, edgecolors='white', 
               linewidths=2, label='Optimal Point', marker='*', zorder=5)
   
   plt.title(f'Optimization Surface: {metric_name}\n(β={BETA_SCORE} F-Score Optimization)', 
            pad=15, fontweight='bold')
   plt.xlabel('Confidence Threshold (trigger if < x)', fontsize=12)
   plt.ylabel('Entropy Threshold (trigger if > y)', fontsize=12)
   plt.legend(loc='upper right', framealpha=0.9)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   
   plt.savefig(filename, bbox_inches='tight', dpi=300)
   plt.close()
   print(f"   [PLOT] Saved: {filename}")

def plot_pareto_frontier(results, ent_grid, conf_grid, output_path):
   """
   Pareto frontier: Precision vs Recall tradeoff.
   Fondamentale per discussione nella tesi.
   """
   plt.figure(figsize=(10, 7))
   
   # Estrai tutte le combinazioni valide
   precision_flat = results["precision"].flatten()
   recall_flat = results["recall"].flatten()
   
   # Rimuovi punti (0,0)
   valid = (precision_flat > 0) | (recall_flat > 0)
   precision_flat = precision_flat[valid]
   recall_flat = recall_flat[valid]
   
   # Scatter plot
   plt.scatter(recall_flat, precision_flat, alpha=0.3, s=20, c='blue', label='All configurations')
   
   # Identifica Pareto frontier (manualmente o con libreria)
   # Qui usiamo un approccio semplice: ordina per recall e trova massima precision
   sorted_indices = np.argsort(recall_flat)
   pareto_recall = []
   pareto_precision = []
   max_precision = 0
   
   for idx in sorted_indices:
      if precision_flat[idx] >= max_precision:
         pareto_recall.append(recall_flat[idx])
         pareto_precision.append(precision_flat[idx])
         max_precision = precision_flat[idx]
   
   plt.plot(pareto_recall, pareto_precision, 'r-', linewidth=2, label='Pareto Frontier', zorder=5)
   
   plt.xlabel('Recall (Error Detection Rate)', fontsize=12)
   plt.ylabel('Precision (Trigger Accuracy)', fontsize=12)
   plt.title('Precision-Recall Tradeoff\n(Pareto-Optimal Configurations)', fontsize=14, fontweight='bold')
   plt.legend(loc='lower left')
   plt.grid(True, alpha=0.3)
   plt.xlim(0, 1.05)
   plt.ylim(0, 1.05)
   plt.tight_layout()
   
   plt.savefig(output_path, bbox_inches='tight', dpi=300)
   plt.close()
   print(f"   [PLOT] Saved: {output_path}")

def generate_latex_table(result: OptimizationResult, output_path: str):
   """
   Genera tabella LaTeX pronta per copia-incolla nella tesi.
   """
   latex_code = r"""\begin{table}[h]
\centering
\caption{Optimal Trigger Configuration Results}
\label{tab:optimization_results}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{95\% CI} \\
\midrule
Entropy Threshold & $> """ + f"{result.ent_threshold:.3f}" + r"""$ & - \\
Confidence Threshold & $< """ + f"{result.conf_threshold:.3f}" + r"""$ & - \\
\midrule
Precision & """ + f"{result.precision:.3f}" + r""" & """ + f"[{result.precision_ci[0]:.3f}, {result.precision_ci[1]:.3f}]" + r""" \\
Recall & """ + f"{result.recall:.3f}" + r""" & """ + f"[{result.recall_ci[0]:.3f}, {result.recall_ci[1]:.3f}]" + r""" \\
$F_{""" + f"{BETA_SCORE}" + r"""}$-Score & """ + f"{result.f_score:.3f}" + r""" & """ + f"[{result.f_score_ci[0]:.3f}, {result.f_score_ci[1]:.3f}]" + r""" \\
\midrule
Intervention Rate & """ + f"{result.intervention_rate:.2%}" + r""" & - \\
Expected Cost & \$""" + f"{result.expected_cost:.2f}" + r""" & - \\
Cost Savings & \$""" + f"{result.cost_savings:.2f}" + r""" (""" + f"{(result.cost_savings/(result.expected_cost+result.cost_savings))*100:.1f}" + r"""\%) & - \\
\midrule
Total Tokens & """ + f"{result.n_samples:,}" + r""" & - \\
Errors Detected & """ + f"{result.true_positives:,}" + r"""/""" + f"{result.n_errors:,}" + r""" & - \\
\bottomrule
\end{tabular}
\end{table}"""
   
   with open(output_path, 'w') as f:
      f.write(latex_code)
   
   print(f"   [LaTeX] Table saved to: {output_path}")

def optimize():
   print(f"="*70)
   print("   ACADEMIC HYPERPARAMETER OPTIMIZATION ENGINE")
   print(f"   Seed: {RANDOM_SEED} | β: {BETA_SCORE} | CI: {CONFIDENCE_LEVEL*100:.0f}%")
   print(f"="*70)
   
   ensure_dir(OUTPUT_DIR)
   set_style()
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"\n   Device: {device}")
   
   # 1. Load
   print("\n[1/6] Loading model and data...")
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
   model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
   model.eval()
   dataset = load_from_disk(DATA_PATH)["validation"]
   
   if SAMPLE_LIMIT:
      print(f"   ⚠️ ATTENZIONE: Usando solo {SAMPLE_LIMIT} campioni!")
      print(f"   Per la tesi, esegui con SAMPLE_LIMIT=None per validità statistica.")
      dataset = dataset.select(range(SAMPLE_LIMIT))
   
   # 2. Inference
   print("\n[2/6] Running inference...")
   entropies, confidences, is_errors = collect_inference_stats_batched(model, dataset, device)
   n_errors = np.sum(is_errors)
   n_total = len(is_errors)
   error_rate = n_errors / n_total
   
   print(f"   Tokens analizzati: {n_total:,}")
   print(f"   Errori trovati: {n_errors:,} ({error_rate:.2%})")
   print(f"   Distribuzione entropy: μ={np.mean(entropies):.3f}, σ={np.std(entropies):.3f}")
   print(f"   Distribuzione confidence: μ={np.mean(confidences):.3f}, σ={np.std(confidences):.3f}")
   
   # 3. Grid Search
   print("\n[3/6] Grid search optimization...")
   ent_grid, conf_grid, results = vectorized_grid_search(entropies, confidences, is_errors)
   
   # 4. Trova ottimo
   print("\n[4/6] Identifying optimal configuration...")
   max_idx = np.unravel_index(np.argmax(results["f_score"]), results["f_score"].shape)
   
   best_ent = ent_grid[max_idx[0]]
   best_conf = conf_grid[max_idx[1]]
   
   # Applica le soglie ottimali
   trigger_mask = (entropies > best_ent) & (confidences < best_conf)
   
   # Calcola metriche
   tp = np.sum(trigger_mask & is_errors)
   fp = np.sum(trigger_mask & ~is_errors)
   fn = np.sum(~trigger_mask & is_errors)
   tn = np.sum(~trigger_mask & ~is_errors)
   
   precision = tp / (tp + fp) if (tp + fp) > 0 else 0
   recall = tp / (tp + fn) if (tp + fn) > 0 else 0
   f_score = results["f_score"][max_idx]
   intervention_rate = np.sum(trigger_mask) / n_total
   
   baseline_cost = results["baseline_cost"]
   expected_cost = results["expected_cost"][max_idx]
   cost_savings = baseline_cost - expected_cost
   
   # 5. Bootstrap CI
   print("\n[5/6] Computing confidence intervals (bootstrap)...")
   
   def precision_fn(triggers, errors):
      tp = np.sum(triggers & errors)
      return tp / np.sum(triggers) if np.sum(triggers) > 0 else 0
   
   def recall_fn(triggers, errors):
      tp = np.sum(triggers & errors)
      return tp / np.sum(errors) if np.sum(errors) > 0 else 0
   
   def fscore_fn(triggers, errors):
      p = precision_fn(triggers, errors)
      r = recall_fn(triggers, errors)
      if p + r == 0:
         return 0
      beta_sq = BETA_SCORE ** 2
      return (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)
   
   precision_ci = bootstrap_metric(trigger_mask, is_errors, precision_fn)
   recall_ci = bootstrap_metric(trigger_mask, is_errors, recall_fn)
   f_score_ci = bootstrap_metric(trigger_mask, is_errors, fscore_fn)
   
   # 6. Crea oggetto risultato
   result = OptimizationResult(
      ent_threshold=best_ent,
      conf_threshold=best_conf,
      precision=precision,
      recall=recall,
      f_score=f_score,
      intervention_rate=intervention_rate,
      expected_cost=expected_cost,
      cost_savings=cost_savings,
      precision_ci=precision_ci,
      recall_ci=recall_ci,
      f_score_ci=f_score_ci,
      n_samples=n_total,
      n_errors=n_errors,
      n_triggers=int(np.sum(trigger_mask)),
      true_positives=int(tp),
      false_positives=int(fp),
      false_negatives=int(fn)
   )
   
   # 7. Output
   print("\n" + "="*70)
   print("   RISULTATI FINALI (PUBLICATION-READY)")
   print("="*70)
   print(f"\n   Soglie Ottimali:")
   print(f"   • Entropy:    > {best_ent:.3f}")
   print(f"   • Confidence: < {best_conf:.3f}")
   print(f"\n   Metriche di Performance:")
   print(f"   • Precision:  {precision:.3f} (95% CI: [{precision_ci[0]:.3f}, {precision_ci[1]:.3f}])")
   print(f"   • Recall:     {recall:.3f} (95% CI: [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}])")
   print(f"   • F{BETA_SCORE}-Score:  {f_score:.3f} (95% CI: [{f_score_ci[0]:.3f}, {f_score_ci[1]:.3f}])")
   print(f"\n   Metriche Economiche:")
   print(f"   • Intervention Rate: {intervention_rate:.2%}")
   print(f"   • Expected Cost:     ${expected_cost:.2f}")
   print(f"   • Cost Savings:      ${cost_savings:.2f} ({(cost_savings/baseline_cost)*100:.1f}%)")
   print(f"\n   Confusion Matrix:")
   print(f"   • True Positives:  {tp:,} (errori rilevati correttamente)")
   print(f"   • False Positives: {fp:,} (falsi allarmi)")
   print(f"   • False Negatives: {fn:,} (errori mancati)")
   print(f"   • True Negatives:  {tn:,}")
   print("="*70)
   
   # 8. Plots
   print("\n[6/6] Generating publication-quality plots...")
   plot_heatmap(ent_grid, conf_grid, results["f_score"], max_idx, 
               f"F{BETA_SCORE}-Score", f"{OUTPUT_DIR}/opt_fscore_heatmap.png")
   plot_heatmap(ent_grid, conf_grid, results["precision"], max_idx, 
               "Precision", f"{OUTPUT_DIR}/opt_precision_heatmap.png")
   plot_heatmap(ent_grid, conf_grid, results["expected_cost"], max_idx,
               "Expected Cost ($)", f"{OUTPUT_DIR}/opt_cost_heatmap.png")
   plot_pareto_frontier(results, ent_grid, conf_grid, f"{OUTPUT_DIR}/pareto_frontier.png")
   
   # 9. Export
   # JSON per riproducibilità (converti NumPy types a Python nativi)
   def convert_to_native(obj):
      """Converte NumPy types a Python nativi per JSON."""
      if isinstance(obj, np.integer):
         return int(obj)
      elif isinstance(obj, np.floating):
         return float(obj)
      elif isinstance(obj, np.ndarray):
         return obj.tolist()
      elif isinstance(obj, tuple):
         return tuple(convert_to_native(item) for item in obj)
      return obj
   
   result_dict = asdict(result)
   result_dict_native = {k: convert_to_native(v) for k, v in result_dict.items()}
   
   with open(f"{OUTPUT_DIR}/results.json", "w") as f:
      json.dump(result_dict_native, f, indent=2)
   
   # TXT leggibile
   with open(f"{OUTPUT_DIR}/optimal_params.txt", "w") as f:
      f.write(f"# Optimal Hyperparameters (F{BETA_SCORE}-Score Optimization)\n")
      f.write(f"# Random Seed: {RANDOM_SEED}\n")
      f.write(f"# Dataset Size: {n_total:,} tokens\n\n")
      for key, value in asdict(result).items():
         f.write(f"{key}: {value}\n")
   
   # LaTeX table
   generate_latex_table(result, f"{OUTPUT_DIR}/results_table.tex")
   
   print(f"\n   ✓ All results saved to: {OUTPUT_DIR}/")
   print(f"   ✓ LaTeX table: results_table.tex (copy-paste nella tesi)")
   print(f"   ✓ JSON export: results.json (per riproducibilità)")
   
   return result

if __name__ == "__main__":
   result = optimize()