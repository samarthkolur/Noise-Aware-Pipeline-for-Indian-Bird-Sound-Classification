"""Paired statistical tests on per-sample correctness (1 = correct, 0 = incorrect)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import stats


def paired_tests(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> Dict[str, Any]:
    """Paired t-test and Wilcoxon signed-rank on correctness vectors (same length)."""
    x = np.asarray(correct_a, dtype=np.float64)
    y = np.asarray(correct_b, dtype=np.float64)
    assert x.shape == y.shape

    diff = y - x
    # Paired t-test: H0 mean(diff)=0; tests whether B differs from A on average
    tt = stats.ttest_rel(y, x)

    wilcoxon_result: Dict[str, Any] = {}
    try:
        # zero_method='wilcox' handles zeros; alternative two-sided
        wr = stats.wilcoxon(y, x, zero_method="wilcox", mode="auto")
        wilcoxon_result = {
            "statistic": float(wr.statistic),
            "pvalue": float(wr.pvalue),
        }
    except ValueError as e:
        wilcoxon_result = {"error": str(e), "statistic": None, "pvalue": None}

    mean_diff = float(np.mean(diff))
    return {
        "comparison": f"{name_b} vs {name_a}",
        "mean_accuracy_delta": mean_diff,
        "n_samples": int(len(x)),
        "paired_ttest": {
            "statistic": float(tt.statistic),
            "pvalue": float(tt.pvalue),
        },
        "wilcoxon_signed_rank": wilcoxon_result,
        "interpretation_note": "p < 0.05 suggests a statistically significant difference in paired correctness.",
    }
