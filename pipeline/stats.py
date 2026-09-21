"""Statistical validation: paired t-test + Wilcoxon signed-rank on per-segment
correctness (design.md §9)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_significance(correctness_a: np.ndarray, correctness_b: np.ndarray) -> dict:
    """Paired t-test + Wilcoxon signed-rank comparing two systems' per-segment
    correctness arrays (design.md: Baseline -> MLP, MLP -> MLP+AE)."""
    correctness_a = np.asarray(correctness_a, dtype=np.float64)
    correctness_b = np.asarray(correctness_b, dtype=np.float64)

    if len(correctness_a) != len(correctness_b):
        raise ValueError("correctness arrays must be the same length (paired per-segment)")

    diffs = correctness_b - correctness_a
    if np.allclose(diffs, 0):
        t_stat, t_p = 0.0, 1.0
        w_stat, w_p = 0.0, 1.0
    else:
        t_stat, t_p = stats.ttest_rel(correctness_b, correctness_a)
        try:
            w_stat, w_p = stats.wilcoxon(correctness_b, correctness_a)
        except ValueError:
            # All differences are zero after ties are dropped.
            w_stat, w_p = 0.0, 1.0

    return {
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(w_p),
        "significant_at_0.05": bool(t_p < 0.05 and w_p < 0.05),
    }
