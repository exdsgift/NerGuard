"""
Mini-batch test of the routable NerGuard pipeline on NVIDIA/Nemotron-PII dataset.

Runs baseline model vs hybrid (model + Ollama LLM router) on a small number
of samples, printing detailed per-sample routing decisions so you can see
exactly what the system does.

Usage:
    uv run python -m src.scripts.nvidia_minibatch_ollama
    uv run python -m src.scripts.nvidia_minibatch_ollama --samples 30 --model gpt-oss:20b
"""

import argparse
import ast
import json
import logging
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.core.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_MODEL_PATH,
    NVIDIA_TO_MODEL_MAP,
)
from src.core.label_mapper import LabelMapper
from src.inference.entity_router import EntitySpecificRouter
from src.inference.llm_router import LLMRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Suppress noisy library logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logger = logging.getLogger("MiniBatch")

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def parse_spans(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return []
    if isinstance(raw, list):
        return sorted(
            [s for s in raw if isinstance(s, dict) and {"label", "start", "end"} <= s.keys()],
            key=lambda x: x["start"],
        )
    return []


def tokenize_and_align(text, spans, tokenizer, mapper, context_length=512):
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=context_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = tokenized["offset_mapping"][0].tolist()
    labels = [mapper.model_label2id.get("O", 0)] * len(offset_mapping)

    for idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            labels[idx] = -100
            continue
        for span in spans:
            if end <= span["start"] or start >= span["end"]:
                continue
            is_start = start == span["start"]
            if not is_start and idx > 0 and labels[idx - 1] != -100:
                prev_lbl = mapper.model_id2label.get(labels[idx - 1], "O")
                curr_base = NVIDIA_TO_MODEL_MAP.get(span["label"], "O")
                if curr_base not in prev_lbl:
                    is_start = True
            labels[idx] = mapper.get_token_label_id(span["label"], is_start)
            break

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "offset_mapping": offset_mapping,
        "labels": labels,
    }


def run_minibatch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(device).eval()
    mapper = LabelMapper(model.config.id2label)
    logger.info("Model loaded.")

    # ── Load Ollama router ───────────────────────────────────────────────────
    logger.info(f"Initializing Ollama router with model: {args.ollama_model}")
    try:
        llm_router = LLMRouter(
            source="ollama",
            ollama_model=args.ollama_model,
            enable_cache=True,
        )
        logger.info(color("Ollama router ready.", GREEN))
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        sys.exit(1)

    entity_router = EntitySpecificRouter(
        entropy_threshold=args.entropy_threshold,
        confidence_threshold=args.confidence_threshold,
        enable_selective=not args.no_selective,
        block_continuation_tokens=not args.allow_continuation,
    )

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info(f"Loading {args.samples} samples from NVIDIA/Nemotron-PII...")
    ds = load_dataset("nvidia/Nemotron-PII", split="train")
    ds = ds.select(range(min(len(ds), args.samples)))
    logger.info(f"Loaded {len(ds)} samples.\n")

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_tokens = 0
    total_correct_base = 0
    total_correct_hybrid = 0
    total_llm_calls = 0
    total_llm_helped = 0
    total_llm_hurt = 0

    for sample_idx, example in enumerate(ds):
        text = example["text"]
        spans = parse_spans(example["spans"])

        print("\n" + "═" * 80)
        print(color(f"  SAMPLE {sample_idx + 1}/{len(ds)}", BOLD))
        print(color(f"  Text (first 120 chars): {text[:120].replace(chr(10), ' ')}...", DIM))
        print("═" * 80)

        processed = tokenize_and_align(text, spans, tokenizer, mapper)
        input_ids = processed["input_ids"].to(device)
        attention_mask = processed["attention_mask"].to(device)
        offset_mapping = processed["offset_mapping"]
        labels_list = processed["labels"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy_tensor = -torch.sum(probs * log_probs, dim=-1)
            confidence_tensor, preds_tensor = torch.max(probs, dim=-1)

        preds_np = preds_tensor[0].cpu().numpy()
        labels_np = labels_list
        offset_np = offset_mapping

        hybrid_preds = list(preds_np)
        prev_label = "O"

        routing_events = []

        for idx in range(len(preds_np)):
            if labels_np[idx] == -100:
                continue

            token_start, token_end = offset_np[idx]
            token_text = text[token_start:token_end]
            pred_id = int(preds_np[idx])
            pred_label = model.config.id2label[pred_id]
            true_id = labels_np[idx]
            true_label = model.config.id2label[true_id]

            conf = float(confidence_tensor[0, idx])
            ent = float(entropy_tensor[0, idx])

            total_tokens += 1
            if pred_id == true_id:
                total_correct_base += 1

            # Check routing
            should_route = entity_router.should_route(
                pred_label, entropy=ent, confidence=conf
            )

            if should_route and llm_router:
                total_llm_calls += 1
                result = llm_router.disambiguate(
                    target_token=token_text,
                    full_text=text,
                    char_start=token_start,
                    char_end=token_end,
                    current_pred=pred_label,
                    prev_label=prev_label,
                )
                corrected_label = result["corrected_label"]
                corrected_id = mapper.model_label2id.get(corrected_label, pred_id)
                cached = result.get("cached", False)

                changed = corrected_id != pred_id
                helped = changed and corrected_id == true_id and pred_id != true_id
                hurt = changed and corrected_id != true_id and pred_id == true_id

                if helped:
                    total_llm_helped += 1
                if hurt:
                    total_llm_hurt += 1

                hybrid_preds[idx] = corrected_id

                routing_events.append({
                    "token": token_text,
                    "true": true_label,
                    "baseline": pred_label,
                    "llm": corrected_label,
                    "conf": conf,
                    "ent": ent,
                    "changed": changed,
                    "helped": helped,
                    "hurt": hurt,
                    "cached": cached,
                    "reasoning": result.get("reasoning", "")[:80],
                })

                hybrid_label = corrected_label
            else:
                hybrid_label = pred_label

            if hybrid_preds[idx] == true_id:
                total_correct_hybrid += 1

            prev_label = hybrid_label

        # ── Print routing events for this sample ──────────────────────────────
        active_mask = [i for i, l in enumerate(labels_np) if l != -100]
        sample_tokens = len(active_mask)
        sample_base_correct = sum(
            1 for i in active_mask if preds_np[i] == labels_np[i]
        )
        sample_hybrid_correct = sum(
            1 for i in active_mask if hybrid_preds[i] == labels_np[i]
        )

        if routing_events:
            print(color(f"  LLM routing events ({len(routing_events)}):", CYAN))
            for ev in routing_events:
                cached_tag = color("[CACHE]", DIM) if ev["cached"] else ""
                status = (
                    color("HELPED", GREEN) if ev["helped"]
                    else color("HURT", RED) if ev["hurt"]
                    else color("no change", DIM) if not ev["changed"]
                    else color("changed", YELLOW)
                )
                print(
                    f"    token={color(repr(ev['token']), BOLD):<20s} "
                    f"true={color(ev['true'], CYAN):<22s} "
                    f"base={ev['baseline']:<22s} "
                    f"llm={color(ev['llm'], YELLOW):<22s} "
                    f"→ {status} {cached_tag}"
                )
                if ev["reasoning"]:
                    print(color(f"      reasoning: {ev['reasoning']}", DIM))
        else:
            print(color("  No LLM routing triggered for this sample.", DIM))

        delta = sample_hybrid_correct - sample_base_correct
        delta_str = (
            color(f"+{delta}", GREEN) if delta > 0
            else color(str(delta), RED) if delta < 0
            else color("0", DIM)
        )
        print(
            f"  Tokens: {sample_tokens} | "
            f"Baseline correct: {sample_base_correct} | "
            f"Hybrid correct: {sample_hybrid_correct} | "
            f"Delta: {delta_str}"
        )

    # ── Global summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print(color("  GLOBAL SUMMARY", BOLD))
    print("═" * 80)
    base_acc = total_correct_base / total_tokens * 100 if total_tokens else 0
    hybrid_acc = total_correct_hybrid / total_tokens * 100 if total_tokens else 0
    delta_acc = hybrid_acc - base_acc

    print(f"  Samples evaluated   : {len(ds)}")
    print(f"  Total active tokens : {total_tokens}")
    print(f"  Baseline accuracy   : {base_acc:.2f}%")
    print(f"  Hybrid accuracy     : {hybrid_acc:.2f}%")
    print(f"  Delta               : {color(f'{delta_acc:+.2f}%', GREEN if delta_acc >= 0 else RED)}")
    print(f"  LLM calls           : {total_llm_calls}")
    print(f"  LLM helped          : {color(str(total_llm_helped), GREEN)}")
    print(f"  LLM hurt            : {color(str(total_llm_hurt), RED)}")
    print(f"  Net corrections     : {color(str(total_llm_helped - total_llm_hurt), GREEN if total_llm_helped >= total_llm_hurt else RED)}")

    cache_stats = llm_router.get_cache_stats()
    if cache_stats:
        print(f"  Cache hit rate      : {cache_stats['hit_rate']} ({cache_stats['hits']} hits / {cache_stats['misses']} misses)")
    print("═" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mini-batch NVIDIA test with Ollama routing"
    )
    parser.add_argument(
        "--samples", type=int, default=20,
        help="Number of samples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH,
        help=f"Path to the NER model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--ollama-model", default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)"
    )
    parser.add_argument(
        "--entropy-threshold", type=float, default=DEFAULT_ENTROPY_THRESHOLD,
        help=f"Entropy threshold for routing (default: {DEFAULT_ENTROPY_THRESHOLD})"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for routing (default: {DEFAULT_CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--no-selective", action="store_true",
        help="Disable entity-specific selective routing (route all uncertain tokens)"
    )
    parser.add_argument(
        "--allow-continuation", action="store_true",
        help="Allow I- continuation tokens to be routed (default: blocked)"
    )
    args = parser.parse_args()
    run_minibatch(args)
