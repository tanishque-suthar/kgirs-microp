"""Statistical analysis module for RAG evaluation.

Computes:
- Spearman and Kendall tau-b correlations with 1000-resample bootstrap 95% CIs.
- Classification metrics: accuracy, quadratic-weighted Cohen's kappa, confusion matrix.
- Position bias flip rates.
- Verbosity bias correlations and fluff-padding delta statistics.
- Multi-rater agreement: pairwise Cohen's kappa and Fleiss' kappa across Rater A, B, and C.
- Retrieval correlation vs quality labels.
"""

import warnings
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RANDOM_SEED

LABEL_MAPPING = {
    "incorrect": 0,
    "partly": 1,
    "correct": 2,
}
ORDERED_CLASSES = ["incorrect", "partly", "correct"]


def encode_labels(series: pd.Series) -> np.ndarray:
    """Map string labels ('incorrect', 'partly', 'correct') to ordinals 0, 1, 2."""
    return series.astype(str).str.lower().map(LABEL_MAPPING).fillna(1).astype(int).to_numpy()


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_func,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """
    Compute point estimate and non-parametric bootstrap (1000 resamples) 95% confidence interval.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(x)
    if n < 3:
        return (0.0, 0.0, 0.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point = float(stat_func(x, y))
            if np.isnan(point):
                point = 0.0
    except Exception:
        point = 0.0

    rng = np.random.default_rng(seed)
    boot_vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_resamples):
            indices = rng.integers(0, n, size=n)
            bx, by = x[indices], y[indices]
            try:
                val = stat_func(bx, by)
                if not np.isnan(val):
                    boot_vals.append(val)
            except Exception:
                pass

    if len(boot_vals) < 50:
        return (point, point, point)

    ci_lower = float(np.percentile(boot_vals, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return (round(point, 4), round(ci_lower, 4), round(ci_upper, 4))


def compute_correlations(
    scores: pd.Series,
    labels: pd.Series,
    metric_name: str = "metric"
) -> Dict[str, Any]:
    """
    Compute Spearman and Kendall tau-b correlations between continuous scores
    and ordinal human labels, with 95% bootstrap CIs.
    """
    valid = (~scores.isna()) & (~labels.isna())
    clean_scores = scores[valid].to_numpy(dtype=float)
    clean_labels = encode_labels(labels[valid])

    if len(clean_scores) < 3:
        return {
            "metric": metric_name,
            "spearman": 0.0,
            "spearman_ci": (0.0, 0.0),
            "spearman_p": 1.0,
            "kendall": 0.0,
            "kendall_ci": (0.0, 0.0),
            "kendall_p": 1.0,
        }

    # Spearman
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp_res = stats.spearmanr(clean_scores, clean_labels)
        sp_p = float(sp_res.pvalue) if (hasattr(sp_res, "pvalue") and not np.isnan(sp_res.pvalue)) else 1.0

    sp_point, sp_low, sp_high = bootstrap_ci(
        clean_scores,
        clean_labels,
        lambda x, y: stats.spearmanr(x, y)[0]
    )

    # Kendall tau-b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kd_res = stats.kendalltau(clean_scores, clean_labels)
        kd_p = float(kd_res.pvalue) if (hasattr(kd_res, "pvalue") and not np.isnan(kd_res.pvalue)) else 1.0

    kd_point, kd_low, kd_high = bootstrap_ci(
        clean_scores,
        clean_labels,
        lambda x, y: stats.kendalltau(x, y)[0]
    )

    return {
        "metric": metric_name,
        "n": len(clean_scores),
        "spearman": 0.0 if np.isnan(sp_point) else sp_point,
        "spearman_ci": (0.0 if np.isnan(sp_low) else sp_low, 0.0 if np.isnan(sp_high) else sp_high),
        "spearman_p": round(sp_p, 5),
        "kendall": 0.0 if np.isnan(kd_point) else kd_point,
        "kendall_ci": (0.0 if np.isnan(kd_low) else kd_low, 0.0 if np.isnan(kd_high) else kd_high),
        "kendall_p": round(kd_p, 5),
    }


def compute_judge_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute classification metrics: Accuracy, Quadratic-weighted Cohen's Kappa,
    Confusion Matrix, and Position Bias Flip Rate.
    """
    if "derived_label" not in df.columns or "judge_verdict" not in df.columns:
        return {}

    y_true_str = df["derived_label"].astype(str).str.lower()
    y_pred_str = df["judge_verdict"].astype(str).str.lower()

    y_true_ord = encode_labels(y_true_str)
    y_pred_ord = encode_labels(y_pred_str)

    acc = accuracy_score(y_true_str, y_pred_str)
    # Quadratic weighted kappa accounts for ordinal distance
    labels_ord = [0, 1, 2]
    kappa_quad = cohen_kappa_score(y_true_ord, y_pred_ord, labels=labels_ord, weights="quadratic")
    kappa_linear = cohen_kappa_score(y_true_ord, y_pred_ord, labels=labels_ord, weights="linear")
    kappa_unweighted = cohen_kappa_score(y_true_str, y_pred_str, labels=ORDERED_CLASSES)

    if np.isnan(kappa_quad):
        kappa_quad = 0.0
    if np.isnan(kappa_linear):
        kappa_linear = 0.0
    if np.isnan(kappa_unweighted):
        kappa_unweighted = 0.0

    cm = confusion_matrix(y_true_str, y_pred_str, labels=ORDERED_CLASSES)

    # Position bias flip rate
    flip_rate = 0.0
    total_judged = len(df)
    if "position_flipped" in df.columns:
        flip_rate = float(df["position_flipped"].mean())
    elif "judge_verdict_forward" in df.columns and "judge_verdict_reversed" in df.columns:
        flips = (df["judge_verdict_forward"].str.lower() != df["judge_verdict_reversed"].str.lower())
        flip_rate = float(flips.mean())

    return {
        "n": total_judged,
        "accuracy": round(float(acc), 4),
        "cohen_kappa_quadratic": round(float(kappa_quad), 4),
        "cohen_kappa_linear": round(float(kappa_linear), 4),
        "cohen_kappa_unweighted": round(float(kappa_unweighted), 4),
        "confusion_matrix": cm.tolist(),
        "classes": ORDERED_CLASSES,
        "position_bias_flip_rate": round(float(flip_rate), 4),
    }


def compute_verbosity_bias(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute verbosity bias metrics:
    1. Spearman correlation between judge score and answer length within each human-label class.
    2. Fluff-padding effect on score delta.
    """
    results = {"by_class": {}, "overall": {}}

    if "answer_length" not in df.columns or "judge_verdict" not in df.columns:
        return results

    df = df.copy()
    df["judge_score"] = encode_labels(df["judge_verdict"])

    # Overall correlation between length and judge score
    if len(df) >= 3:
        if df["judge_score"].nunique() > 1 and df["answer_length"].nunique() > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp_overall = stats.spearmanr(df["answer_length"], df["judge_score"])
                stat = float(sp_overall.statistic) if hasattr(sp_overall, "statistic") else float(sp_overall[0])
                pval = float(sp_overall.pvalue) if hasattr(sp_overall, "pvalue") else float(sp_overall[1])
                results["overall"] = {
                    "n": len(df),
                    "spearman": round(stat, 4) if not np.isnan(stat) else 0.0,
                    "p_value": round(pval, 5) if not np.isnan(pval) else 1.0,
                }
        else:
            results["overall"] = {
                "n": len(df),
                "spearman": 0.0,
                "p_value": 1.0,
                "note": "Constant judge scores or lengths",
            }

    # Within-class correlations
    for cls in ORDERED_CLASSES:
        sub = df[df["derived_label"].str.lower() == cls]
        if len(sub) >= 5 and sub["judge_score"].nunique() > 1:
            sp = stats.spearmanr(sub["answer_length"], sub["judge_score"])
            results["by_class"][cls] = {
                "n": len(sub),
                "spearman": round(float(sp.statistic), 4),
                "p_value": round(float(sp.pvalue), 5),
            }
        else:
            results["by_class"][cls] = {
                "n": len(sub),
                "spearman": 0.0,
                "p_value": 1.0,
                "note": "Insufficient variance or sample size",
            }

    # Fluff perturbation analysis
    fluff_df = df[df["is_fluff_tested"] == True] if "is_fluff_tested" in df.columns else pd.DataFrame()
    if len(fluff_df) > 0 and "fluff_score_delta" in fluff_df.columns:
        deltas = fluff_df["fluff_score_delta"].dropna().to_numpy()
        results["fluff_test"] = {
            "n_tested": len(deltas),
            "mean_delta": round(float(np.mean(deltas)), 4) if len(deltas) > 0 else 0.0,
            "pct_unchanged": round(float(np.mean(deltas == 0) * 100), 1) if len(deltas) > 0 else 0.0,
            "pct_increased": round(float(np.mean(deltas > 0) * 100), 1) if len(deltas) > 0 else 0.0,
            "pct_decreased": round(float(np.mean(deltas < 0) * 100), 1) if len(deltas) > 0 else 0.0,
        }
    else:
        results["fluff_test"] = None

    return results


def compute_inter_annotator_agreement(
    derived_series: pd.Series,
    my_series: pd.Series,
    judge_series: pd.Series,
) -> Dict[str, Any]:
    """
    Compute pairwise quadratic-weighted Cohen's kappa and Fleiss' kappa
    across Rater A (RAGTruth-derived), Rater B (User manual), and Rater C (Gemini judge).
    """
    df = pd.DataFrame({
        "rater_a": derived_series.astype(str).str.lower(),
        "rater_b": my_series.astype(str).str.lower(),
        "rater_c": judge_series.astype(str).str.lower(),
    }).dropna()

    # Keep only rows where all 3 raters provided valid ratings
    valid_mask = (
        df["rater_a"].isin(ORDERED_CLASSES)
        & df["rater_b"].isin(ORDERED_CLASSES)
        & df["rater_c"].isin(ORDERED_CLASSES)
    )
    df = df[valid_mask]
    n = len(df)
    if n < 3:
        return {"n": n, "status": "insufficient_data"}

    a_ord = encode_labels(df["rater_a"])
    b_ord = encode_labels(df["rater_b"])
    c_ord = encode_labels(df["rater_c"])

    # Pairwise quadratic-weighted kappas
    labels_ord = [0, 1, 2]
    kappa_ab = cohen_kappa_score(a_ord, b_ord, labels=labels_ord, weights="quadratic")
    kappa_ac = cohen_kappa_score(a_ord, c_ord, labels=labels_ord, weights="quadratic")
    kappa_bc = cohen_kappa_score(b_ord, c_ord, labels=labels_ord, weights="quadratic")

    kappa_ab = 0.0 if np.isnan(kappa_ab) else kappa_ab
    kappa_ac = 0.0 if np.isnan(kappa_ac) else kappa_ac
    kappa_bc = 0.0 if np.isnan(kappa_bc) else kappa_bc

    # Fleiss' Kappa across all 3 raters
    rating_matrix = np.column_stack([a_ord, b_ord, c_ord])
    try:
        agg, cats = aggregate_raters(rating_matrix)
        fk = float(fleiss_kappa(agg))
    except Exception:
        fk = 0.0

    return {
        "n": n,
        "pairwise": {
            "A_vs_B (RAGTruth vs User)": round(float(kappa_ab), 4),
            "A_vs_C (RAGTruth vs Gemini)": round(float(kappa_ac), 4),
            "B_vs_C (User vs Gemini)": round(float(kappa_bc), 4),
        },
        "fleiss_kappa": round(fk, 4),
        "status": "complete",
    }


def compute_retrieval_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Spearman and Kendall correlations between retrieval metrics
    (MRR, NDCG) and ground truth quality labels.
    """
    if "reciprocal_rank" not in df.columns or "derived_label" not in df.columns:
        return {}

    mrr_corr = compute_correlations(df["reciprocal_rank"], df["derived_label"], metric_name="MRR")
    ndcg_corr = compute_correlations(df["ndcg"], df["derived_label"], metric_name="NDCG")

    return {
        "mrr": mrr_corr,
        "ndcg": ndcg_corr,
        "mean_mrr": round(float(df["reciprocal_rank"].mean()), 4),
        "mean_ndcg": round(float(df["ndcg"].mean()), 4),
        "zero_relevant_rate": round(float((df["reciprocal_rank"] == 0).mean()), 4),
    }
