import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import os
import sys
import ollama
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.prompt import PROMPT, VALID_LABELS_STR  # noqa: E402
from src.utils.sample_PII import SAMPLE1  # noqa: E402

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LLM_SOURCE = "openai"
# LLM_SOURCE = "ollama"
# LLM_SOURCE = "None"

OLLAMA_MODEL = "qwen2.5:3b"

# Sliding Window Settings
MAX_LEN = 512
OVERLAP = 128
STRIDE = MAX_LEN - 2 - OVERLAP


class LLMRouter:
    def __init__(self, source: str = LLM_SOURCE):
        self.source = source
        if self.source == "openai":
            if not OPENAI_API_KEY:
                sys.exit("Apy key missing!")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            print(f"   [LLM] Backend: OpenAI ({OPENAI_MODEL})")
        else:
            self.client = None
            print(f"   [LLM] Backend: Ollama ({OLLAMA_MODEL})")

    def disambiguate(
        self,
        target_token: str,
        full_text: str,
        char_start: int,
        char_end: int,
        current_pred: str,
        prev_label: str,
        lang: str,
    ) -> Dict[str, Any]:
        WINDOW_SIZE = 200
        ctx_start = max(0, char_start - WINDOW_SIZE)
        if ctx_start > 0:
            while ctx_start > 0 and full_text[ctx_start] not in [" ", "\n", "."]:
                ctx_start -= 1
            ctx_start += 1

        ctx_end = min(len(full_text), char_end + WINDOW_SIZE)
        if ctx_end < len(full_text):
            while ctx_end < len(full_text) and full_text[ctx_end] not in [
                " ",
                "\n",
                ".",
            ]:
                ctx_end += 1

        prefix = full_text[ctx_start:char_start].replace("\n", " ")
        target_snippet = full_text[
            char_start:char_end
        ]
        suffix = full_text[char_end:ctx_end].replace("\n", " ")

        highlighted_context = f"...{prefix}>>> {target_snippet} <<<{suffix}..."
        clean_token = target_snippet.strip()

        try:
            prompt_content = PROMPT.format(
                context=highlighted_context,
                target_token=clean_token,
                prev_label=prev_label,
                current_pred=current_pred,
                valid_labels_str=VALID_LABELS_STR,
            )
        except KeyError as e:
            print(f"   [PROMPT ERROR] Missing key: {e}")
            return {
                "is_pii": False,
                "corrected_label": current_pred,
                "reasoning": "Error",
            }

        try:
            if self.source == "openai":
                result = self._call_openai(prompt_content)
            else:
                result = self._call_ollama(prompt_content)

            reasoning = result.get("reasoning", "No reasoning provided")

            final_tag = result.get("corrected_label", "O").strip()

            return {
                "is_pii": final_tag != "O",
                "corrected_label": final_tag,
                "reasoning": reasoning,
            }

        except Exception as e:
            print(f"   [LLM ERROR]: {e}")
            return {
                "is_pii": False,
                "corrected_label": current_pred,
                "reasoning": "Error in LLM call",
            }

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},  # Deterministic
        )
        return json.loads(response["message"]["content"])

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                # System prompt allineato al ruolo "Linguistic Expert"
                {
                    "role": "system",
                    "content": "You are a strict PII annotation expert. Output valid JSON only. Do not add conversational text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)


class PIITester:
    def __init__(self, model_path, label_path, llm_routing=True):
        self.device = DEVICE
        self.id2label = self._load_labels(label_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(
            self.device
        )
        self.model.eval()

        self.router = LLMRouter(source=LLM_SOURCE)
        self.llm_routing = llm_routing

    def _load_labels(self, path):
        if not os.path.exists(path):
            sys.exit(f"File label not found: {path}")
        with open(path, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}

    def _visualize_censoring(self, token_results):
        reconstructed = ""
        last_label = "O"

        for res in token_results:
            if res is None:
                continue

            token = res["token"]
            label = res["final_label"]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            is_new_word = token.startswith(" ") or token.startswith("▁")
            clean_token = token.replace(" ", "").replace("▁", "")

            prefix = " " if (is_new_word and reconstructed) else ""

            if label == "O":
                reconstructed += prefix + clean_token
                last_label = "O"
            else:
                current_type = label.split("-")[-1]
                last_type = last_label.split("-")[-1] if last_label != "O" else None

                if label.startswith("I-") and (current_type == last_type):
                    pass
                else:
                    reconstructed += prefix + "█" * 8
                last_label = label

        print(f"\n   [CENSORED PREVIEW]:\n   {reconstructed}")

    def _predict_chunk(self, input_ids):
        tensor_input = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(tensor_input).logits[0]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        confidences, pred_ids = torch.max(probs, dim=-1)
        return confidences, pred_ids, entropy

    def analyze_sentence(self, lang_code, text):
        print(f"\n[{lang_code.upper()}] Processing Text ({len(text)} chars)...")

        all_tokens_encoding = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        all_input_ids = all_tokens_encoding["input_ids"]
        all_offsets = all_tokens_encoding["offset_mapping"]
        total_tokens = len(all_input_ids)

        full_analysis = [None] * total_tokens
        max_model_content = MAX_LEN - 2

        print(f"   Total Tokens: {total_tokens}")
        print(f"   {'TOKEN':<15}\t{'PRED':<15}\t\t{'CONF':<8} {'ENTR':<8} {'SOURCE'}")
        print("   " + "_" * 85)

        if total_tokens <= max_model_content:
            input_ids = (
                [self.tokenizer.cls_token_id]
                + all_input_ids
                + [self.tokenizer.sep_token_id]
            )
            conf, preds, ent = self._predict_chunk(input_ids)

            valid_conf = conf[1:-1]
            valid_preds = preds[1:-1]
            valid_ent = ent[1:-1]
            tokens = self.tokenizer.convert_ids_to_tokens(all_input_ids)

            self._process_tokens(
                start_idx=0,
                tokens=tokens,
                preds=valid_preds,
                conf=valid_conf,
                ent=valid_ent,
                text=text,
                lang_code=lang_code,
                total_tokens=total_tokens,
                buffer=full_analysis,
                offsets=all_offsets,
            )

        else:
            print(
                f"   [INFO] Text exceeds {MAX_LEN} context. Using Sliding Window (Stride: {STRIDE})..."
            )
            for i in range(0, total_tokens, STRIDE):
                chunk_end = min(i + max_model_content, total_tokens)
                chunk_ids = all_input_ids[i:chunk_end]

                input_ids = (
                    [self.tokenizer.cls_token_id]
                    + chunk_ids
                    + [self.tokenizer.sep_token_id]
                )
                conf, preds, ent = self._predict_chunk(input_ids)

                valid_conf = conf[1:-1]
                valid_preds = preds[1:-1]
                valid_ent = ent[1:-1]
                chunk_tokens = self.tokenizer.convert_ids_to_tokens(chunk_ids)

                tokens_to_process = len(chunk_ids)
                if chunk_end < total_tokens:
                    tokens_to_process = min(len(chunk_ids), STRIDE)

                self._process_tokens(
                    start_idx=i,
                    tokens=chunk_tokens[:tokens_to_process],
                    preds=valid_preds[:tokens_to_process],
                    conf=valid_conf[:tokens_to_process],
                    ent=valid_ent[:tokens_to_process],
                    text=text,
                    lang_code=lang_code,
                    total_tokens=total_tokens,
                    buffer=full_analysis,
                    offsets=all_offsets[i : i + tokens_to_process],
                )

        self._visualize_censoring(full_analysis)
        print("\n" + "=" * 100 + "\n")

    def _process_tokens(
        self,
        start_idx,
        tokens,
        preds,
        conf,
        ent,
        text,
        lang_code,
        total_tokens,
        buffer,
        offsets,
    ):
        for j, token in enumerate(tokens):
            global_idx = start_idx + j

            pred_id = preds[j].item()
            model_label = self.id2label.get(pred_id, "UNK")
            confidence = conf[j].item()
            entropy = ent[j].item()

            final_label = model_label
            source = "Model"
            char_start, char_end = offsets[j]

            # Prev Label Logic
            prev_label = "O"
            if global_idx > 0:
                prev_item = buffer[global_idx - 1]
                if prev_item is not None:
                    prev_label = prev_item["final_label"]
            if self.llm_routing:
                if entropy > THRESHOLD and confidence < THRESHOLD_CONF:
                    source = f"LLM ({self.router.source})"

                    llm_result = self.router.disambiguate(
                        target_token=token,
                        full_text=text,
                        char_start=char_start,
                        char_end=char_end,
                        current_pred=model_label,
                        prev_label=prev_label,
                        lang=lang_code,
                    )
                if llm_result.get("is_pii"):
                    final_label = llm_result.get("corrected_label", model_label)
                else:
                    final_label = "O"

            buffer[global_idx] = {"token": token, "final_label": final_label}

            print(
                f"   {token:<15}\t{final_label:<15.15}\t\t{confidence:.4f}   {entropy:.4f}   {source}"
            )


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

MODEL_PATH = "./models/mdeberta-pii-safe/final"
LABEL_PATH = "./data/processed/id2label.json"
THRESHOLD = 0.60
THRESHOLD_CONF = 0.85

if __name__ == "__main__":
    tester = PIITester(MODEL_PATH, LABEL_PATH, llm_routing=False)

    long_sample = SAMPLE1

    tester.analyze_sentence("en", long_sample)
