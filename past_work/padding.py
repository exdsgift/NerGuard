from transformers import DataCollatorWithPadding
import numpy as np
from typing import Dict, List, Union
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

import torch


class NERCollatorSoftmax:
    def __init__(self, tokenizer, label_pad_id=-100, pad_to_multiple_of=32):
        self.inner = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=pad_to_multiple_of
        )
        self.label_pad_id = label_pad_id

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.inner(inputs)
        max_len = batch["input_ids"].shape[1]
        padded = [y + [self.label_pad_id] * (max_len - len(y)) for y in labels]
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


# def compute_metrics_softmax(eval_pred, id2label=None):
#     if id2label is None:
#         from training import id2label
#     pred_logits, label_ids = eval_pred   # [B,T,K], [B,T]
#     pred_ids = pred_logits.argmax(-1)

#     y_true, y_pred = [], []
#     for yt, yp in zip(label_ids, pred_ids):
#         true_tags = [id2label[int(i)] for i in yt if int(i) != -100]
#         pred_tags = [id2label[int(i)] for i in yp[:len(true_tags)]]
#         y_true.append(true_tags); y_pred.append(pred_tags)

#     f1 = f1_score(y_true, y_pred, scheme=IOB2, mode="strict")
#     # opzionale: print(classification_report(y_true, y_pred, scheme=IOB2, mode="strict"))
#     return {"f1": float(f1)}


def compute_metrics_softmax(eval_pred, id2label: Union[Dict[int, str], List[str]]):
    """
    eval_pred: (pred_logits, label_ids) come passato dal Trainer
      - pred_logits: [B, T, K] (np.ndarray o torch.Tensor) oppure tuple(..., )
      - label_ids:   [B, T] (np.ndarray o torch.Tensor) con -100 sui token da ignorare
    id2label: mapping id->tag (dict o list), es: {0:'O', 1:'B-PER', 2:'I-PER', ...}
    """
    pred_logits, label_ids = eval_pred

    # 1) Estrai logits (se vengono come tuple) e portali a numpy
    if isinstance(pred_logits, tuple):
        pred_logits = pred_logits[0]
    try:
        import torch

        if isinstance(pred_logits, torch.Tensor):
            pred_ids = pred_logits.detach().cpu().argmax(-1).numpy()
        else:
            pred_ids = np.asarray(pred_logits).argmax(-1)
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.detach().cpu().numpy()
        else:
            label_ids = np.asarray(label_ids)
    except Exception:
        # fallback puro numpy
        pred_ids = np.asarray(pred_logits).argmax(-1)
        label_ids = np.asarray(label_ids)

    # 2) Costruisci sequenze di tag (seqeval vuole liste di liste di stringhe)
    def _lab(i: int) -> str:
        # id2label può essere list o dict
        return id2label[int(i)] if isinstance(id2label, dict) else id2label[int(i)]

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    # token-accuracy (sul subset non ignorato)
    mask_valid = label_ids != -100
    total_tokens = int(mask_valid.sum())
    correct_tokens = int(((pred_ids == label_ids) & mask_valid).sum())

    for i in range(label_ids.shape[0]):
        mask = mask_valid[i]
        true_ids_i = label_ids[i][mask]
        pred_ids_i = pred_ids[i][mask]
        y_true.append([_lab(x) for x in true_ids_i.tolist()])
        y_pred.append([_lab(x) for x in pred_ids_i.tolist()])

    # 3) Metriche a livello ENTITÀ (seqeval, strict IOB2)
    prec = precision_score(y_true, y_pred, scheme=IOB2, mode="strict")
    rec = recall_score(y_true, y_pred, scheme=IOB2, mode="strict")
    f1 = f1_score(y_true, y_pred, scheme=IOB2, mode="strict")

    # 4) Ritorna float “puri” (evita warning di gather su tensori scalari)
    out = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "token_accuracy": float(correct_tokens / total_tokens)
        if total_tokens > 0
        else 0.0,
    }
    return out
