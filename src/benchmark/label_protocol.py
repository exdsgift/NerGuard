"""Label integrity protocol: intersection logic, overlap documentation.

Implements the Tier 1 + Tier 2 strategy:
- Tier 1: strict native label intersection (always computed)
- Tier 2: optional semantic alignment from JSON file (per system-dataset pair)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class LabelOverlapReport:
    """Documents the label overlap between a system and a dataset."""

    system_name: str
    dataset_name: str
    system_native_labels: Set[str]
    dataset_native_labels: Set[str]
    evaluated_labels: Set[str]
    system_excluded: Set[str]  # System labels not in dataset
    dataset_excluded: Set[str]  # Dataset labels not in system
    mapping_applied: bool
    semantic_alignment: Optional[Dict[str, List[str]]] = None
    tier: int = 1  # 1 = strict native, 2 = semantic alignment

    def to_dict(self) -> Dict:
        return {
            "system_name": self.system_name,
            "dataset_name": self.dataset_name,
            "system_native_labels": sorted(self.system_native_labels),
            "dataset_native_labels": sorted(self.dataset_native_labels),
            "evaluated_labels": sorted(self.evaluated_labels),
            "excluded_labels": {
                "from_system": sorted(self.system_excluded),
                "from_dataset": sorted(self.dataset_excluded),
            },
            "mapping_applied": self.mapping_applied,
            "semantic_alignment": self.semantic_alignment,
            "tier": self.tier,
            "n_evaluated": len(self.evaluated_labels),
            "n_system_total": len(self.system_native_labels),
            "n_dataset_total": len(self.dataset_native_labels),
        }


def compute_label_overlap(
    system_name: str,
    dataset_name: str,
    system_labels: Set[str],
    dataset_labels: Set[str],
    semantic_alignment: Optional[Dict[str, List[str]]] = None,
) -> LabelOverlapReport:
    """Compute the evaluable label set between a system and dataset.

    Tier 1 (semantic_alignment=None):
        evaluated_labels = system_labels ∩ dataset_labels (exact string match)

    Tier 2 (semantic_alignment provided):
        The alignment dict maps system labels to dataset labels.
        e.g., {"PERSON": ["GIVENNAME", "SURNAME"]} means the system's PERSON
        label is semantically equivalent to the dataset's GIVENNAME/SURNAME.
        Evaluated labels = all dataset labels that have a system equivalent.

    Args:
        system_name: Name of the system.
        dataset_name: Name of the dataset.
        system_labels: Set of native entity labels from the system (no BIO prefix).
        dataset_labels: Set of native entity labels from the dataset (no BIO prefix).
        semantic_alignment: Optional mapping from system labels to dataset labels.

    Returns:
        LabelOverlapReport with full documentation.
    """
    if semantic_alignment is None:
        # Tier 1: strict intersection
        evaluated = system_labels & dataset_labels
        system_excluded = system_labels - dataset_labels
        dataset_excluded = dataset_labels - system_labels

        report = LabelOverlapReport(
            system_name=system_name,
            dataset_name=dataset_name,
            system_native_labels=system_labels,
            dataset_native_labels=dataset_labels,
            evaluated_labels=evaluated,
            system_excluded=system_excluded,
            dataset_excluded=dataset_excluded,
            mapping_applied=False,
            semantic_alignment=None,
            tier=1,
        )
    else:
        # Tier 2: semantic alignment
        # Build the set of dataset labels that can be evaluated
        evaluated_dataset_labels = set()
        for sys_label, ds_labels in semantic_alignment.items():
            if sys_label in system_labels:
                for dl in ds_labels:
                    if dl in dataset_labels:
                        evaluated_dataset_labels.add(dl)

        # System labels that participate
        evaluated_system_labels = {
            sl for sl, dls in semantic_alignment.items()
            if sl in system_labels and any(dl in dataset_labels for dl in dls)
        }

        system_excluded = system_labels - evaluated_system_labels
        dataset_excluded = dataset_labels - evaluated_dataset_labels

        report = LabelOverlapReport(
            system_name=system_name,
            dataset_name=dataset_name,
            system_native_labels=system_labels,
            dataset_native_labels=dataset_labels,
            evaluated_labels=evaluated_dataset_labels,
            system_excluded=system_excluded,
            dataset_excluded=dataset_excluded,
            mapping_applied=True,
            semantic_alignment=semantic_alignment,
            tier=2,
        )

    # Log summary
    tier_str = f"Tier {report.tier}"
    logger.info(
        f"[{tier_str}] {system_name} × {dataset_name}: "
        f"{len(report.evaluated_labels)} evaluable labels "
        f"(sys={len(system_labels)}, ds={len(dataset_labels)})"
    )
    if not report.evaluated_labels:
        logger.warning(
            f"  ZERO overlap — this pair will produce no metrics. "
            f"Consider providing semantic alignment (Tier 2)."
        )

    return report


def load_semantic_alignment(
    alignment_path: str,
    system_name: str,
    dataset_name: str,
) -> Optional[Dict[str, List[str]]]:
    """Load semantic alignment for a specific system-dataset pair.

    The alignment JSON has structure:
    {
        "system_key": {
            "dataset_key": {
                "SYSTEM_LABEL": ["DATASET_LABEL_1", "DATASET_LABEL_2"],
                ...
            }
        }
    }

    Args:
        alignment_path: Path to the alignment JSON file.
        system_name: System identifier (e.g., "spacy").
        dataset_name: Dataset identifier (e.g., "ai4privacy").

    Returns:
        Alignment dict for this pair, or None if not found.
    """
    path = Path(alignment_path)
    if not path.exists():
        logger.warning(f"Alignment file not found: {path}")
        return None

    with open(path) as f:
        all_alignments = json.load(f)

    # Normalize keys for lookup
    sys_key = system_name.lower().replace(" ", "-").replace("_", "-")
    ds_key = dataset_name.lower().replace(" ", "-").replace("_", "-")

    sys_alignments = all_alignments.get(sys_key, {})
    pair_alignment = sys_alignments.get(ds_key)

    if pair_alignment is None:
        logger.debug(f"No alignment found for {sys_key} × {ds_key}")
        return None

    logger.info(f"Loaded Tier 2 alignment for {sys_key} × {ds_key}: {len(pair_alignment)} mappings")
    return pair_alignment


def build_tier2_label_map(
    semantic_alignment: Dict[str, List[str]],
) -> Dict[str, str]:
    """Build a reverse map: dataset_label -> system_label.

    Used to remap ground truth labels to system labels for comparison.

    Args:
        semantic_alignment: {system_label: [dataset_label, ...]}

    Returns:
        {dataset_label: system_label}
    """
    reverse = {}
    for sys_label, ds_labels in semantic_alignment.items():
        for dl in ds_labels:
            reverse[dl] = sys_label
    return reverse
