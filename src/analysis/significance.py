"""Statistical significance tests for NER system comparison.

Provides paired bootstrap and McNemar's test for comparing base vs hybrid systems.
"""

import numpy as np
from typing import Dict, List, Tuple

from seqeval.metrics import f1_score as seqeval_f1
from seqeval.scheme import IOB2


def _per_sample_entity_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> np.ndarray:
    """Compute entity-level F1 per sample using seqeval."""
    scores = []
    for yt, yp in zip(y_true, y_pred):
        try:
            f1 = seqeval_f1([yt], [yp], mode="strict", scheme=IOB2, zero_division=0)
        except Exception:
            f1 = seqeval_f1([yt], [yp], zero_division=0)
        scores.append(f1)
    return np.array(scores)


def paired_bootstrap_test(
    y_true: List[List[str]],
    y_pred_a: List[List[str]],
    y_pred_b: List[List[str]],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap significance test on per-sample entity F1.

    Tests whether system B is significantly better than system A.

    Returns:
        dict with keys: p_value, mean_delta, ci_lower, ci_upper, n_samples
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    scores_a = _per_sample_entity_f1(y_true, y_pred_a)
    scores_b = _per_sample_entity_f1(y_true, y_pred_b)
    observed_delta = scores_b.mean() - scores_a.mean()

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = scores_b[idx].mean() - scores_a[idx].mean()

    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)

    # Two-sided p-value via shifted bootstrap (H0: delta = 0)
    # Shift the bootstrap distribution to be centered at 0
    shifted_deltas = deltas - deltas.mean()
    p_value = np.mean(np.abs(shifted_deltas) >= np.abs(observed_delta))

    return {
        "p_value": float(p_value),
        "mean_delta": float(observed_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
        "scores_a_mean": float(scores_a.mean()),
        "scores_b_mean": float(scores_b.mean()),
    }


def mcnemar_test(
    y_true: List[List[str]],
    y_pred_a: List[List[str]],
    y_pred_b: List[List[str]],
) -> Dict:
    """McNemar's test on token-level correctness.

    Compares discordant pairs: tokens where one system is correct and the other wrong.

    Returns:
        dict with keys: p_value, chi2, n_a_correct_b_wrong, n_b_correct_a_wrong, n_tokens
    """
    from scipy.stats import chi2 as chi2_dist

    # Flatten to token level
    n_a_right_b_wrong = 0  # A correct, B wrong
    n_b_right_a_wrong = 0  # B correct, A wrong
    n_both_right = 0
    n_both_wrong = 0
    n_tokens = 0

    for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b):
        for t, a, b in zip(yt, ya, yb):
            n_tokens += 1
            a_correct = (a == t)
            b_correct = (b == t)
            if a_correct and b_correct:
                n_both_right += 1
            elif a_correct and not b_correct:
                n_a_right_b_wrong += 1
            elif not a_correct and b_correct:
                n_b_right_a_wrong += 1
            else:
                n_both_wrong += 1

    # McNemar's chi-squared with continuity correction
    n_discordant = n_a_right_b_wrong + n_b_right_a_wrong
    if n_discordant == 0:
        return {
            "p_value": 1.0,
            "chi2": 0.0,
            "n_a_correct_b_wrong": n_a_right_b_wrong,
            "n_b_correct_a_wrong": n_b_right_a_wrong,
            "n_both_correct": n_both_right,
            "n_both_wrong": n_both_wrong,
            "n_tokens": n_tokens,
        }

    chi2 = (abs(n_b_right_a_wrong - n_a_right_b_wrong) - 1) ** 2 / n_discordant
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "p_value": float(p_value),
        "chi2": float(chi2),
        "n_a_correct_b_wrong": n_a_right_b_wrong,
        "n_b_correct_a_wrong": n_b_right_a_wrong,
        "n_both_correct": n_both_right,
        "n_both_wrong": n_both_wrong,
        "n_tokens": n_tokens,
    }
