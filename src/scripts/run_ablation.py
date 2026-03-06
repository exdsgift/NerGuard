"""Ablation study: isolate contribution of each routing component.

Runs 5 variants on nvidia-pii:
1. never-route  — entropy=inf (nothing triggers routing)
2. always-route — entropy=-1, confidence=2.0 (everything triggers)
3. entropy-only — confidence=2.0 (only entropy decides)
4. confidence-only — entropy=-1 (only confidence decides)
5. full-system  — calibrated thresholds (default)

Usage:
    uv run python -m src.scripts.run_ablation \
        --samples 1000 --llm-model gpt-4o --batch-llm \
        --semantic-alignment alignments/default.json
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import sys
import time
import tracemalloc

import torch
from tqdm import tqdm

from src.benchmark.datasets.nvidia_pii import NvidiaPIIAdapter
from src.benchmark.label_protocol import (
    compute_label_overlap,
    load_semantic_alignment,
)
from src.benchmark.metrics import MetricsComputer
from src.benchmark.systems.base import SystemPrediction
from src.core.route_config import RouteConfig
from src.inference.entity_router import EntitySpecificRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Ablation")

ABLATION_VARIANTS = {
    "never_route": RouteConfig(
        entropy_threshold=float("inf"),
        confidence_threshold=0.0,
    ),
    "always_route": RouteConfig(
        entropy_threshold=-1.0,
        confidence_threshold=2.0,
    ),
    "entropy_only": RouteConfig(
        entropy_threshold=0.583,  # will be overridden by calibration
        confidence_threshold=2.0,  # disabled
    ),
    "confidence_only": RouteConfig(
        entropy_threshold=-1.0,  # disabled
        confidence_threshold=0.787,  # will be overridden by calibration
    ),
    "full_system": None,  # uses calibrated defaults
}


def parse_args():
    p = argparse.ArgumentParser(description="NerGuard Ablation Study")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--calibrate", type=int, default=200)
    p.add_argument("--llm-model", type=str, default="gpt-4o")
    p.add_argument("--llm-source", type=str, default="openai")
    p.add_argument("--batch-llm", action="store_true")
    p.add_argument("--batch-llm-concurrency", type=int, default=50)
    p.add_argument("--semantic-alignment", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="experiments/ablation_nvidia-pii")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--model-path", type=str, default="./models/mdeberta-pii-safe/final")
    return p.parse_args()


def run_variant(
    variant_name: str,
    route_config_override: RouteConfig | None,
    system,
    samples,
    calibration_samples,
    alignment,
    metrics_computer: MetricsComputer,
    args,
    output_dir: str,
):
    """Run a single ablation variant."""
    variant_dir = os.path.join(output_dir, variant_name)
    results_path = os.path.join(variant_dir, "results.json")

    # Skip if already done
    if os.path.exists(results_path):
        logger.info(f"  Skipping {variant_name} (already exists)")
        with open(results_path) as f:
            return json.load(f)

    os.makedirs(variant_dir, exist_ok=True)

    # Apply routing configuration
    if variant_name == "full_system":
        # Calibrate thresholds normally
        if calibration_samples:
            system.calibrate_thresholds(calibration_samples)
            logger.info(f"  [{variant_name}] Calibrated thresholds")
        # else: use defaults from setup()
    elif variant_name == "entropy_only":
        # Calibrate first, then override confidence to disable it
        if calibration_samples:
            system.calibrate_thresholds(calibration_samples)
        calibrated_entropy = system.entity_router.entropy_threshold
        config = RouteConfig(
            entropy_threshold=calibrated_entropy,
            confidence_threshold=2.0,  # disabled
        )
        system.entity_router = EntitySpecificRouter.from_config(config)
        logger.info(f"  [{variant_name}] entropy={calibrated_entropy:.4f}, confidence=disabled")
    elif variant_name == "confidence_only":
        # Calibrate first, then override entropy to disable it
        if calibration_samples:
            system.calibrate_thresholds(calibration_samples)
        calibrated_confidence = system.entity_router.confidence_threshold
        config = RouteConfig(
            entropy_threshold=-1.0,  # disabled
            confidence_threshold=calibrated_confidence,
        )
        system.entity_router = EntitySpecificRouter.from_config(config)
        logger.info(f"  [{variant_name}] entropy=disabled, confidence={calibrated_confidence:.4f}")
    else:
        # never_route or always_route: use fixed config
        system.entity_router = EntitySpecificRouter.from_config(route_config_override)
        logger.info(f"  [{variant_name}] entropy={route_config_override.entropy_threshold}, confidence={route_config_override.confidence_threshold}")

    # Run inference
    ds_labels = set()
    for s in samples:
        for l in s.labels:
            if l != "O":
                ds_labels.add(l.replace("B-", "").replace("I-", ""))

    sys_labels = system.native_labels()
    overlap = compute_label_overlap(
        system_name=variant_name,
        dataset_name="nvidia-pii",
        system_labels=sys_labels,
        dataset_labels=ds_labels,
        semantic_alignment=alignment,
    )

    use_batch = (
        args.batch_llm
        and hasattr(system, "predict_ner_only")
        and hasattr(system, "resolve_routing_batch")
    )

    tracemalloc.start()

    y_pred_samples = []
    latencies = []

    if use_batch:
        deferred = []
        for idx, sample in enumerate(tqdm(samples, desc=f"  {variant_name} [NER]", leave=False)):
            try:
                d = system.predict_ner_only(idx, sample.text, sample.tokens, sample.token_spans)
                deferred.append(d)
            except Exception as e:
                logger.debug(f"NER failed: {e}")

        total_pending = sum(len(d.pending_spans) for d in deferred)
        logger.info(f"  [{variant_name}] NER done: {len(deferred)} samples, {total_pending} spans pending")

        loop = asyncio.new_event_loop()
        try:
            predictions = loop.run_until_complete(
                system.resolve_routing_batch(deferred, max_concurrent=args.batch_llm_concurrency)
            )
        finally:
            loop.close()

        for pred in predictions:
            y_pred_samples.append(pred.labels)
            latencies.append(pred.latency_ms)
    else:
        for sample in tqdm(samples, desc=f"  {variant_name}", leave=False):
            try:
                pred = system.predict(sample.text, sample.tokens, sample.token_spans)
                y_pred_samples.append(pred.labels)
                latencies.append(pred.latency_ms)
            except Exception as e:
                y_pred_samples.append(["O"] * len(sample.tokens))
                latencies.append(0.0)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_true_samples = [s.labels for s in samples]
    metrics = metrics_computer.compute_all(
        y_true_samples=y_true_samples,
        y_pred_samples=y_pred_samples,
        overlap_report=overlap,
        latencies=latencies,
        texts=[s.text for s in samples],
        sample_ids=[s.sample_id for s in samples],
        tokens_per_sample=[s.tokens for s in samples],
    )
    metrics.memory_peak_mb = peak / (1024 * 1024)

    # Get routing metadata
    routing_meta = {}
    if hasattr(system, "get_routing_metadata"):
        routing_meta = system.get_routing_metadata()

    result = {
        "variant": variant_name,
        "dataset": "nvidia-pii",
        "n_samples": len(samples),
        "n_evaluated_labels": len(overlap.evaluated_labels),
        "token_level": {
            "f1_macro": metrics.f1_macro,
            "f1_micro": metrics.f1_micro,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
        },
        "entity_level": {
            "precision": metrics.entity_precision,
            "recall": metrics.entity_recall,
            "f1": metrics.entity_f1,
        },
        "latency": {
            "mean_ms": metrics.latency_mean_ms,
            "median_ms": metrics.latency_median_ms,
        },
        "routing": routing_meta,
    }

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save aligned predictions for significance tests
    with open(os.path.join(variant_dir, "predictions.json"), "w") as f:
        json.dump({
            "y_true": metrics.aligned_y_true,
            "y_pred": metrics.aligned_y_pred,
        }, f)

    logger.info(
        f"  [{variant_name}] F1-macro={metrics.f1_macro:.4f} | "
        f"Entity-F1={metrics.entity_f1:.4f} | "
        f"LLM calls={routing_meta.get('llm_calls', 0)}"
    )

    return result


def main():
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()
    output_dir = f"{args.output_dir}_{args.samples}samples"

    logger.info(f"Ablation study: {args.samples} samples on nvidia-pii")
    logger.info(f"Output: {output_dir}")

    # Load dataset
    adapter = NvidiaPIIAdapter()
    total_needed = args.samples + args.calibrate
    all_samples = adapter.load(max_samples=total_needed, seed=args.seed)

    calibration_samples = all_samples[:args.calibrate]
    eval_samples = all_samples[args.calibrate:]
    logger.info(f"Loaded {len(eval_samples)} eval + {len(calibration_samples)} calibration samples")

    # Load alignment
    alignment = None
    if args.semantic_alignment:
        alignment = load_semantic_alignment(args.semantic_alignment, "nerguard", "nvidia-pii")

    metrics_computer = MetricsComputer()

    # Create and setup system once
    from src.benchmark.systems.nerguard_hybrid_v2 import NerGuardHybridV2
    system = NerGuardHybridV2(
        model_path=args.model_path,
        device=args.device,
        llm_source=args.llm_source,
        llm_model=args.llm_model,
    )
    system.setup()

    # Set dataset labels for O-span routing
    ds_labels = adapter.native_labels()
    if hasattr(system, "set_dataset_labels"):
        system.set_dataset_labels(ds_labels)

    all_results = []

    for variant_name, route_config in ABLATION_VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Variant: {variant_name}")
        logger.info(f"{'='*60}")

        result = run_variant(
            variant_name=variant_name,
            route_config_override=route_config,
            system=system,
            samples=eval_samples,
            calibration_samples=calibration_samples,
            alignment=alignment,
            metrics_computer=metrics_computer,
            args=args,
            output_dir=output_dir,
        )
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY — nvidia-pii ({args.samples} samples)")
    print(f"{'='*80}")
    print(f"{'Variant':<20} {'F1-macro':>10} {'Entity-F1':>10} {'Precision':>10} {'Recall':>10} {'LLM calls':>10}")
    print(f"{'-'*80}")

    for r in all_results:
        tl = r.get("token_level", {})
        el = r.get("entity_level", {})
        llm = r.get("routing", {}).get("llm_calls", 0)
        print(
            f"{r['variant']:<20} {tl.get('f1_macro', 0):>10.4f} "
            f"{el.get('f1', 0):>10.4f} {el.get('precision', 0):>10.4f} "
            f"{el.get('recall', 0):>10.4f} {llm:>10}"
        )

    # Save summary
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    system.teardown()


if __name__ == "__main__":
    main()
