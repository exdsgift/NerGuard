"""Main benchmark runner — orchestrates datasets, systems, metrics, and reports."""

import gc
import logging
import os
import sys
import tracemalloc
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.benchmark.config import BenchmarkConfig, parse_args
from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter
from src.benchmark.label_protocol import (
    LabelOverlapReport,
    compute_label_overlap,
    load_semantic_alignment,
)
from src.benchmark.metrics import BenchmarkMetrics, MetricsComputer
from src.benchmark.report import generate_experiment_output, generate_summary
from src.benchmark.systems.base import SystemWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BenchmarkRunner")


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.metrics_computer = MetricsComputer()
        self.all_results = []
        # Session directory groups all experiments from this run
        self.session_dir = os.path.join(self.config.output_dir, self.timestamp)

    def run(self) -> None:
        logger.info(f"NER PII Benchmark starting at {self.timestamp}")
        logger.info(f"Systems: {self.config.systems}")
        logger.info(f"Datasets: {self.config.datasets}")
        logger.info(f"Output: {self.session_dir}")

        os.makedirs(self.session_dir, exist_ok=True)

        # Load all datasets first (they persist across systems)
        loaded_datasets: Dict[str, Tuple[DatasetAdapter, List[BenchmarkSample]]] = {}
        for ds_name in self.config.datasets:
            adapter = self._create_dataset_adapter(ds_name)
            if adapter is None:
                continue
            max_samples = self.config.get_samples_for_dataset(ds_name)
            logger.info(f"Loading dataset: {ds_name} (max_samples={max_samples or 'all'})")
            samples = adapter.load(
                max_samples=max_samples,
                seed=self.config.seed,
                languages=self.config.languages,
            )
            loaded_datasets[ds_name] = (adapter, samples)
            logger.info(f"  Loaded {len(samples)} samples")

        # Run each system sequentially (setup → evaluate on all datasets → teardown)
        for sys_name in self.config.systems:
            wrapper = self._create_system_wrapper(sys_name)
            if wrapper is None:
                logger.warning(f"Skipping unknown system: {sys_name}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"System: {wrapper.name()}")
            logger.info(f"{'='*60}")

            try:
                wrapper.setup()
            except Exception as e:
                logger.error(f"Failed to setup {sys_name}: {e}")
                continue

            for ds_name, (adapter, samples) in loaded_datasets.items():
                self._run_combination(wrapper, adapter, samples, ds_name)

            wrapper.teardown()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate comparative summary
        if self.all_results:
            generate_summary(self.all_results, self.session_dir, self.timestamp)

        # Print final summary table
        self._print_summary()

    def _run_combination(
        self,
        system: SystemWrapper,
        adapter: DatasetAdapter,
        samples: List[BenchmarkSample],
        ds_name: str,
    ) -> None:
        sys_name = system.name()

        # Check idempotency — skip if results already exist
        sys_clean = sys_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        ds_clean = ds_name.lower().replace(" ", "_")
        exp_dir = os.path.join(self.session_dir, f"{sys_clean}_{ds_clean}")
        logger.debug(f"  Experiment dir: {exp_dir}")
        if os.path.exists(os.path.join(exp_dir, "results.json")):
            logger.info(f"  Skipping {sys_name} × {ds_name} (already exists)")
            return

        logger.info(f"\n  Evaluating: {sys_name} × {ds_name} ({len(samples)} samples)")

        # For GLiNER: set dataset labels dynamically
        if hasattr(system, "set_dataset_labels"):
            system.set_dataset_labels(adapter.native_labels())

        # Compute label overlap
        alignment = None
        if self.config.semantic_alignment:
            alignment = load_semantic_alignment(
                self.config.semantic_alignment,
                sys_name.lower().split()[0],  # first word as key
                ds_name,
            )

        # For Tier 2 evaluation of systems on datasets with native labels
        # If alignment is None, we check if system labels overlap with dataset labels
        sys_labels = system.native_labels()
        ds_labels = adapter.native_labels()

        overlap = compute_label_overlap(
            system_name=sys_name,
            dataset_name=ds_name,
            system_labels=sys_labels,
            dataset_labels=ds_labels,
            semantic_alignment=alignment,
        )

        if not overlap.evaluated_labels:
            logger.warning(f"  ZERO label overlap — skipping metric computation (Tier 1)")
            logger.warning(f"    System labels: {sorted(sys_labels)[:10]}...")
            logger.warning(f"    Dataset labels: {sorted(ds_labels)[:10]}...")
            # Still save config for documentation
            empty_metrics = BenchmarkMetrics(n_samples=len(samples))
            generate_experiment_output(
                system_name=sys_name,
                dataset_name=ds_name,
                metrics=empty_metrics,
                overlap_report=overlap,
                config=self._config_dict(),
                output_dir=self.session_dir,
                timestamp=self.timestamp,
            )
            return

        # Run inference
        y_pred_samples = []
        latencies = []

        tracemalloc.start()
        gpu_mem_before = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_mem_before = torch.cuda.max_memory_allocated()

        # Two-pass batch mode for hybrid systems with LLM routing
        use_batch = (
            self.config.batch_llm
            and hasattr(system, "predict_ner_only")
            and hasattr(system, "resolve_routing_batch")
        )

        if use_batch:
            y_pred_samples, latencies = self._run_batch_inference(
                system, samples, sys_name
            )
        else:
            for sample in tqdm(samples, desc=f"    {sys_name}", leave=False):
                try:
                    pred = system.predict(sample.text, sample.tokens, sample.token_spans)
                    y_pred_samples.append(pred.labels)
                    latencies.append(pred.latency_ms)
                except Exception as e:
                    logger.debug(f"    Prediction failed for {sample.sample_id}: {e}")
                    y_pred_samples.append(["O"] * len(sample.tokens))
                    latencies.append(0.0)

        # Memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peak_mb = peak / (1024 * 1024)
        gpu_peak_mb = 0.0
        if torch.cuda.is_available():
            gpu_peak_mb = (torch.cuda.max_memory_allocated() - gpu_mem_before) / (1024 * 1024)

        # Compute metrics
        y_true_samples = [s.labels for s in samples]
        metrics = self.metrics_computer.compute_all(
            y_true_samples=y_true_samples,
            y_pred_samples=y_pred_samples,
            overlap_report=overlap,
            latencies=latencies,
            texts=[s.text for s in samples],
            sample_ids=[s.sample_id for s in samples],
            tokens_per_sample=[s.tokens for s in samples],
        )
        metrics.memory_peak_mb = memory_peak_mb
        metrics.gpu_peak_mb = gpu_peak_mb

        # Log summary
        logger.info(
            f"    F1-macro={metrics.f1_macro:.4f} | Entity-F1={metrics.entity_f1:.4f} | "
            f"Latency={metrics.latency_mean_ms:.1f}ms | "
            f"Labels={len(overlap.evaluated_labels)} (Tier {overlap.tier})"
        )

        # Save output
        generate_experiment_output(
            system_name=sys_name,
            dataset_name=ds_name,
            metrics=metrics,
            overlap_report=overlap,
            config=self._config_dict(),
            output_dir=self.session_dir,
            timestamp=self.timestamp,
        )

        # Collect for summary
        self.all_results.append({
            "system": sys_name,
            "dataset": ds_name,
            "f1_macro": metrics.f1_macro,
            "f1_micro": metrics.f1_micro,
            "entity_f1": metrics.entity_f1,
            "latency_mean_ms": metrics.latency_mean_ms,
            "n_evaluated_labels": len(overlap.evaluated_labels),
            "tier": overlap.tier,
            "n_samples": metrics.n_samples,
        })

    def _run_batch_inference(
        self,
        system,
        samples: List,
        sys_name: str,
    ) -> Tuple[List[List[str]], List[float]]:
        """Two-pass batch inference for hybrid systems with LLM routing.

        Pass 1: NER-only inference on all samples (fast, GPU)
        Pass 2: Batch all LLM calls with async concurrency
        Pass 3: Apply corrections and finalize predictions
        """
        import asyncio

        # Pass 1: NER-only
        deferred = []
        failed_indices = set()
        for idx, sample in enumerate(tqdm(samples, desc=f"    {sys_name} [NER]", leave=False)):
            try:
                d = system.predict_ner_only(idx, sample.text, sample.tokens, sample.token_spans)
                deferred.append(d)
            except Exception as e:
                logger.debug(f"    NER failed for {sample.sample_id}: {e}")
                failed_indices.add(idx)

        total_pending = sum(len(d.pending_spans) for d in deferred)
        logger.info(f"    NER done: {len(deferred)} samples, {total_pending} spans pending LLM routing")

        if total_pending == 0:
            # No LLM calls needed — finalize directly
            predictions = []
            for d in deferred:
                from src.benchmark.systems.base import SystemPrediction
                predictions.append(SystemPrediction(labels=["O"] * len(d.tokens), latency_ms=d.ner_latency_ms))
            # Still need to run resolve to apply regex post-processing
            loop = asyncio.new_event_loop()
            try:
                predictions = loop.run_until_complete(
                    system.resolve_routing_batch(deferred, max_concurrent=1)
                )
            finally:
                loop.close()
        else:
            # Pass 2+3: Batch LLM routing
            loop = asyncio.new_event_loop()
            try:
                predictions = loop.run_until_complete(
                    system.resolve_routing_batch(
                        deferred,
                        max_concurrent=self.config.batch_llm_concurrency,
                    )
                )
            finally:
                loop.close()

        # Build output lists (handle failed samples)
        y_pred_samples = []
        latencies = []
        pred_iter = iter(predictions)
        for idx, sample in enumerate(samples):
            if idx in failed_indices:
                y_pred_samples.append(["O"] * len(sample.tokens))
                latencies.append(0.0)
            else:
                pred = next(pred_iter)
                y_pred_samples.append(pred.labels)
                latencies.append(pred.latency_ms)

        return y_pred_samples, latencies

    def _create_dataset_adapter(self, name: str) -> Optional[DatasetAdapter]:
        if name == "ai4privacy":
            from src.benchmark.datasets.ai4privacy import AI4PrivacyAdapter
            return AI4PrivacyAdapter()
        elif name == "nvidia-pii":
            from src.benchmark.datasets.nvidia_pii import NvidiaPIIAdapter
            return NvidiaPIIAdapter()
        elif name == "wikineural":
            from src.benchmark.datasets.wikineural import WikiNeuralAdapter
            return WikiNeuralAdapter(languages=self.config.languages)
        else:
            logger.warning(f"Unknown dataset: {name}")
            return None

    def _create_system_wrapper(self, name: str) -> Optional[SystemWrapper]:
        if name == "nerguard-base":
            from src.benchmark.systems.nerguard_base import NerGuardBase
            return NerGuardBase(model_path=self.config.model_path, device=self.config.device)
        elif name == "nerguard-hybrid":
            from src.benchmark.systems.nerguard_hybrid import NerGuardHybrid
            return NerGuardHybrid(
                model_path=self.config.model_path,
                device=self.config.device,
                llm_source=self.config.llm_source,
                llm_model=self.config.llm_model,
            )
        elif name == "nerguard-hybrid-v2":
            from src.benchmark.systems.nerguard_hybrid_v2 import NerGuardHybridV2
            return NerGuardHybridV2(
                model_path=self.config.model_path,
                device=self.config.device,
                llm_source=self.config.llm_source,
                llm_model=self.config.llm_model,
            )
        elif name == "piiranha":
            from src.benchmark.systems.piiranha import PiiranhaWrapper
            return PiiranhaWrapper(device=self.config.device)
        elif name == "piiranha-hybrid":
            from src.benchmark.systems.piiranha_hybrid import PiiranhaHybridWrapper
            return PiiranhaHybridWrapper(
                device=self.config.device,
                llm_source=self.config.llm_source,
                llm_model=self.config.llm_model,
            )
        elif name == "presidio":
            from src.benchmark.systems.presidio_sys import PresidioWrapper
            return PresidioWrapper()
        elif name == "gliner":
            from src.benchmark.systems.gliner_sys import GLiNERWrapper
            return GLiNERWrapper(device=self.config.device)
        elif name == "spacy":
            from src.benchmark.systems.spacy_ner import SpacyWrapper
            return SpacyWrapper()
        elif name == "bert-ner":
            from src.benchmark.systems.bert_ner import BertNERWrapper
            return BertNERWrapper(device=self.config.device)
        else:
            logger.warning(f"Unknown system: {name}")
            return None

    def _config_dict(self) -> Dict:
        return {
            "seed": self.config.seed,
            "batch_size": self.config.batch_size,
            "runs": self.config.runs,
            "model_path": self.config.model_path,
            "llm_source": self.config.llm_source,
            "llm_model": self.config.llm_model,
            "device": self.config.device,
        }

    def _print_summary(self) -> None:
        if not self.all_results:
            logger.info("No results to summarize.")
            return

        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")

        by_dataset = {}
        for r in self.all_results:
            by_dataset.setdefault(r["dataset"], []).append(r)

        for ds_name, results in by_dataset.items():
            print(f"\nDataset: {ds_name}")
            print(f"{'-'*70}")
            print(f"{'System':<35} {'F1-macro':>10} {'Entity-F1':>10} {'Latency':>10}")
            print(f"{'-'*70}")

            for r in sorted(results, key=lambda x: x.get("f1_macro", 0), reverse=True):
                print(
                    f"{r['system']:<35} {r['f1_macro']:>10.4f} "
                    f"{r['entity_f1']:>10.4f} {r['latency_mean_ms']:>9.1f}ms"
                )

        print(f"\n{'='*80}")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    config = parse_args()
    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
