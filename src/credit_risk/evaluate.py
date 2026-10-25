"""Evaluation metrics used in credit risk.

AUC     - chance the model ranks a random bad loan as riskier than a random good loan (0.5 = coin flip).
Gini    - 2 x AUC - 1. Same information as AUC, but it's the number credit risk teams usually quote.
KS      - the biggest gap between the score distributions of good and bad loans.
Brier   - average squared gap between predicted PD and what happened (lower is better).
Log loss- like Brier, but punishes confident wrong predictions much harder.
Calibration - in groups of similar predicted PD, does the predicted default rate match reality?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             roc_curve)


def gini(y_true, y_score) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def summary(y_true, y_score) -> dict:
    y_true, y_score = np.asarray(y_true), np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "loans": int(len(y_true)),
        "bad_rate": float(y_true.mean()),
        "mean_predicted_pd": float(y_score.mean()),
        "auc": float(roc_auc_score(y_true, y_score)),
        "gini": float(gini(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score)),
    }


def bootstrap_ci(y_true, y_score, metric=roc_auc_score, n_boot: int = 300, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Confidence interval by resampling loans with replacement.

    Caveat: this treats loans as independent. Loans issued in the same month share economic
    conditions, so the true uncertainty is probably a bit wider than this interval.
    """
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n, stats = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() != y_true[idx].max():
            stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_auc_difference(y_true, score_a, score_b, n_boot: int = 300, seed: int = 7) -> dict:
    """Bootstrap the AUC difference between two models scored on the SAME loans."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y_true), np.asarray(score_a), np.asarray(score_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            diffs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"auc_difference": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            "ci_low": float(lo), "ci_high": float(hi)}


def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(y_score)})
    df["group"] = pd.qcut(df["pd"].rank(method="first"), n_bins, labels=False) + 1
    out = (df.groupby("group")
             .agg(loans=("y", "size"), predicted_pd=("pd", "mean"), actual_bad_rate=("y", "mean"))
             .reset_index())
    out["gap"] = out["actual_bad_rate"] - out["predicted_pd"]
    return out


def calibration_slope_intercept(y_true, y_score) -> dict:
    """Fit actual outcome ~ logit(predicted). A well calibrated model has slope ~1 and intercept ~0."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, np.asarray(y_true))
    return {"calibration_slope": float(lr.coef_[0][0]), "calibration_intercept": float(lr.intercept_[0])}


def confusion_at(y_true, y_score, threshold: float) -> dict:
    """Treat PD >= threshold as 'predicted bad'."""
    y, pred = np.asarray(y_true), np.asarray(y_score) >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"threshold": threshold, "true_bad_caught": tp, "good_flagged": fp, "bad_missed": fn,
            "good_passed": tn, "recall_bad": tp / max(tp + fn, 1), "precision_bad": tp / max(tp + fp, 1)}


def metrics_by_period(df: pd.DataFrame, score_col: str, period: str = "Q") -> pd.DataFrame:
    """AUC, bad rate and average predicted PD for each issue quarter (or month)."""
    rows = []
    for key, g in df.groupby(df["issue_date"].dt.to_period(period)):
        if g["bad"].nunique() < 2:
            continue
        rows.append({"period": str(key), "loans": len(g), "bad_rate": g["bad"].mean(),
                     "mean_predicted_pd": g[score_col].mean(), "auc": roc_auc_score(g["bad"], g[score_col])})
    return pd.DataFrame(rows, columns=["period", "loans", "bad_rate", "mean_predicted_pd", "auc"])


def metrics_by_segment(df: pd.DataFrame, score_col: str, segment: str, min_loans: int = 300) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(segment, observed=True):
        if len(g) < min_loans or g["bad"].nunique() < 2:
            continue
        rows.append({segment: key, "loans": len(g), "bad_rate": g["bad"].mean(),
                     "mean_predicted_pd": g[score_col].mean(), "auc": roc_auc_score(g["bad"], g[score_col])})
    cols = [segment, "loans", "bad_rate", "mean_predicted_pd", "auc"]
    return pd.DataFrame(rows, columns=cols).sort_values("loans", ascending=False).reset_index(drop=True)
