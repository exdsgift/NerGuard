import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from datasets import load_from_disk
from sklearn.metrics import classification_report
import os
import logging
from dotenv import load_dotenv

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    LABEL_TO_UNIFIED,
)
from src.inference.entity_router import EntitySpecificRouter

GLINER_AVAILABLE = False
PRESIDIO_AVAILABLE = False
SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from gliner import GLiNER

    GLINER_AVAILABLE = True
except ImportError:
    logger.warning("GLiNER not available, skipping")

try:
    from presidio_analyzer import AnalyzerEngine

    PRESIDIO_AVAILABLE = True
except ImportError:
    logger.warning("Presidio not available, skipping")

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("SpaCy not available, skipping")

try:
    from src.inference.llm_router import LLMRouter

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLMRouter not available, hybrid mode disabled")

MODEL_PATH = DEFAULT_MODEL_PATH
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = "./plots/evaluation_results"
SAMPLE_LIMIT = 3000


@dataclass
class EntitySpan:
    label: str
    start: int
    end: int
    text: str

    @property
    def unified_label(self):
        clean = self.label.upper()
        if "-" in clean and len(clean.split("-")[0]) <= 2:
            clean = clean.split("-", 1)[1]

        if "SSN" in clean or "DRIVER" in clean or "PASSPORT" in clean:
            return "GOV_ID"
        if "MAIL" in clean:
            return "EMAIL_ADDRESS"
        if "PHONE" in clean:
            return "PHONE_NUMBER"

        return LABEL_TO_UNIFIED.get(clean, "O")


def plot_results(results, output_dir="evaluation_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics = []
    for res in results:
        y_true_no_o = [y for y in res["y_true"] if y != "O"]
        y_pred_no_o = [
            res["y_pred"][i] for i, y in enumerate(res["y_true"]) if y != "O"
        ]

        report = classification_report(
            y_true_no_o if y_true_no_o else res["y_true"],
            y_pred_no_o if y_true_no_o else res["y_pred"],
            output_dict=True,
            zero_division=0,
        )

        metrics.append(
            {
                "Model": res["model"],
                "F1-Score": report["macro avg"]["f1-score"],
                "Latency (ms)": res["latency"],
            }
        )

    df_metrics = pd.DataFrame(metrics)

    df_metrics.to_csv(f"{output_dir}/metrics_summary.csv", index=False)


class ModelWrapper:
    def predict(self, sample, tokenizer, id2label) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class NerGuard(ModelWrapper):
    def __init__(self, model_path, enable_llm=True, use_selective_routing=True):
        self.enable_llm = enable_llm and LLM_AVAILABLE
        self.use_selective_routing = use_selective_routing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(
            self.device
        )
        self.model.eval()

        self.router = LLMRouter(source="openai") if self.enable_llm else None
        self.id2label = self.model.config.id2label

        self.thresh_ent = DEFAULT_ENTROPY_THRESHOLD
        self.thresh_conf = DEFAULT_CONFIDENCE_THRESHOLD

        # Entity-specific router for selective routing
        self.entity_router = EntitySpecificRouter(
            entropy_threshold=self.thresh_ent,
            confidence_threshold=self.thresh_conf,
            enable_selective=use_selective_routing,
        )

        self.llm_calls = 0
        self.llm_changed_to_entity = 0
        self.llm_changed_to_o = 0
        self.llm_kept_same = 0
        self.high_entropy_tokens = 0

        if self.enable_llm:
            print(
                f"    LLM enabled with thresholds: entropy > {self.thresh_ent}, conf < {self.thresh_conf}"
            )
            print(f"    Selective routing: {use_selective_routing}")

    def name(self):
        if not self.enable_llm:
            return "NerGuard (Base)"
        elif self.use_selective_routing:
            return "NerGuard (Hybrid+Selective)"
        else:
            return "NerGuard (Hybrid)"

    def predict(self, sample, tokenizer, id2label):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decode text for LLM context
        text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Get offset mapping for LLM routing
        try:
            encoding = tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]
        except Exception:
            encoding = tokenizer.encode_plus(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]

        # Model prediction
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids_tensor, attention_mask=attention_mask).logits[
                0
            ]

        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
        conf, pred_ids = torch.max(probs, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        final_preds = []
        final_gt = []
        prev_pred = "O"

        for i, lbl_id in enumerate(labels):
            if lbl_id == -100:
                continue

            gt_lbl = id2label[lbl_id].replace("B-", "").replace("I-", "").upper()
            final_gt.append(LABEL_TO_UNIFIED.get(gt_lbl, "O"))
            pred_id = pred_ids[i].item()
            raw_pred = self.id2label[pred_id]

            # Use entity-specific routing
            should_route = (
                self.enable_llm
                and self.entity_router.should_route(
                    predicted_label=raw_pred,
                    entropy=entropy[i].item(),
                    confidence=conf[i].item(),
                )
            )

            if should_route:
                self.high_entropy_tokens += 1

                try:
                    self.llm_calls += 1

                    if i < len(offset_mapping):
                        char_start, char_end = offset_mapping[i]

                        res = self.router.disambiguate(
                            target_token=tokens[i],
                            full_text=text,
                            char_start=char_start,
                            char_end=char_end,
                            current_pred=raw_pred,
                            prev_label=prev_pred,
                            lang="en",
                        )

                        # Track changes
                        old_is_entity = raw_pred != "O"
                        new_label = res.get("corrected_label", raw_pred)
                        new_is_entity = new_label != "O"

                        if old_is_entity == new_is_entity:
                            self.llm_kept_same += 1
                        elif not old_is_entity and new_is_entity:
                            self.llm_changed_to_entity += 1
                        elif old_is_entity and not new_is_entity:
                            self.llm_changed_to_o += 1

                        if res.get("is_pii"):
                            raw_pred = new_label
                        else:
                            raw_pred = "O"
                except Exception:
                    pass

            clean_pred = raw_pred.replace("B-", "").replace("I-", "").upper()
            final_preds.append(LABEL_TO_UNIFIED.get(clean_pred, "O"))
            prev_pred = raw_pred

        return final_preds, final_gt


class ExternalWrapper(ModelWrapper):
    def __init__(self, engine_type):
        self.type = engine_type
        if engine_type == "spacy" and SPACY_AVAILABLE:
            if not spacy.util.is_package("en_core_web_lg"):
                spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
        elif engine_type == "presidio" and PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
        elif engine_type == "gliner" and GLINER_AVAILABLE:
            self.model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.labels = [
                "person",
                "city",
                "street",
                "phone",
                "email",
                "date",
                "id",
                "passport",
                "age",
                "gender",
                "job",
            ]
            self.map = {
                "person": "PERSON",
                "city": "LOCATION",
                "street": "LOCATION",
                "phone": "PHONE_NUMBER",
                "email": "EMAIL_ADDRESS",
                "date": "DATE_TIME",
                "id": "GOV_ID",
                "passport": "GOV_ID",
                "age": "AGE",
                "gender": "GENDER",
                "job": "TITLE",
            }

    def name(self):
        return self.type.capitalize()

    def _get_spans(self, text):
        spans = []
        if self.type == "spacy":
            doc = self.nlp(text)
            spans = [
                EntitySpan(ent.label_, ent.start_char, ent.end_char, ent.text)
                for ent in doc.ents
            ]
        elif self.type == "presidio":
            res = self.analyzer.analyze(text=text, language="en")
            spans = [
                EntitySpan(r.entity_type, r.start, r.end, text[r.start : r.end])
                for r in res
            ]
        elif self.type == "gliner":
            preds = self.model.predict_entities(text, self.labels, threshold=0.5)
            spans = [
                EntitySpan(
                    self.map.get(p["label"], "O"), p["start"], p["end"], p["text"]
                )
                for p in preds
            ]
        return spans

    def predict(self, sample, tokenizer, id2label):
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        spans = self._get_spans(text)

        try:
            encoding = tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]
        except Exception:
            encoding = tokenizer.encode_plus(
                text, return_offsets_mapping=True, add_special_tokens=True
            )
            offset_mapping = encoding["offset_mapping"]

        final_preds = []
        final_gt = []

        for i, lbl_id in enumerate(labels):
            if lbl_id == -100:
                continue

            gt_lbl = id2label[lbl_id].replace("B-", "").replace("I-", "").upper()
            final_gt.append(LABEL_TO_UNIFIED.get(gt_lbl, "O"))

            if i < len(offset_mapping):
                start_char, end_char = offset_mapping[i]

                if start_char == end_char == 0:
                    final_preds.append("O")
                    continue

                token_pred = "O"
                for span in spans:
                    overlap_start = max(start_char, span.start)
                    overlap_end = min(end_char, span.end)

                    if overlap_end > overlap_start:
                        overlap_ratio = (overlap_end - overlap_start) / (
                            end_char - start_char
                        )
                        if overlap_ratio > 0.5:
                            token_pred = span.unified_label
                            break

                final_preds.append(token_pred)
            else:
                final_preds.append("O")

        return final_preds, final_gt


# main


def run_benchmark():
    print(f"Loading Data from {DATA_PATH}...")
    dataset = load_from_disk(DATA_PATH)["validation"]
    if SAMPLE_LIMIT:
        dataset = dataset.select(range(min(SAMPLE_LIMIT, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    id2label = AutoConfig.from_pretrained(MODEL_PATH).id2label

    models = []
    models.append(NerGuard(MODEL_PATH, enable_llm=False))

    if LLM_AVAILABLE:
        # Hybrid with selective routing (entity-specific)
        models.append(NerGuard(MODEL_PATH, enable_llm=True, use_selective_routing=True))
        # Optionally also test without selective routing for comparison
        # models.append(NerGuard(MODEL_PATH, enable_llm=True, use_selective_routing=False))
    else:
        print("   - Skipping NerGuard Hybrid (LLMRouter not available)\n")

    if SPACY_AVAILABLE:
        models.append(ExternalWrapper("spacy"))
    if PRESIDIO_AVAILABLE:
        models.append(ExternalWrapper("presidio"))
    if GLINER_AVAILABLE:
        models.append(ExternalWrapper("gliner"))

    results = []
    print(f"- Benchmarking {len(models)} models...\n")

    for model in models:
        print(f"- {model.name()}")
        y_true, y_pred, lats = [], [], []

        for sample in tqdm(dataset, desc=f"  Processing", leave=False):
            t0 = time.time()
            try:
                p, g = model.predict(sample, tokenizer, id2label)

                if len(p) == len(g) and len(p) > 0:
                    y_pred.extend(p)
                    y_true.extend(g)
                    lats.append((time.time() - t0) * 1000)
            except Exception as e:
                pass

        avg_lat = np.mean(lats) if lats else 0

        llm_info = ""
        if isinstance(model, NerGuard) and model.enable_llm:
            total_tokens = len(
                [l for sample in dataset for l in sample["labels"] if l != -100]
            )
            llm_info = f"\n    High entropy tokens: {model.high_entropy_tokens}/{total_tokens} ({100 * model.high_entropy_tokens / total_tokens:.1f}%)"
            llm_info += f"\n    LLM calls: {model.llm_calls}"
            llm_info += f"\n    Changes: O→Entity={model.llm_changed_to_entity}, Entity→O={model.llm_changed_to_o}, Same={model.llm_kept_same}"

        print(f"  Done: {len(lats)} samples | {avg_lat:.2f}ms{llm_info}\n")

        results.append(
            {
                "model": model.name(),
                "y_true": y_true,
                "y_pred": y_pred,
                "latency": avg_lat,
            }
        )

    plot_results(results)
    print(f"Done! Results in {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    load_dotenv()
    run_benchmark()
