"""Monitoring a model after it goes live, replayed on history.

Two kinds of monitoring:
  * Available straight away (no outcomes needed): score and feature distributions, missing values.
    Measured with the Population Stability Index (PSI) against the training data.
  * Needs outcomes to mature (here: up to 36 months): AUC, calibration and bad rates by vintage.

PSI rule of thumb: below 0.10 little change, 0.10-0.25 worth investigating, above 0.25 a large
shift. These are conventions, not laws; the thresholds are configurable and a high PSI is a
prompt to investigate, not an automatic rebuild.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4) -> float:
    """PSI between a reference sample and a new sample. Bins are deciles of the reference."""
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().to_numpy()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual, eps: float = 1e-4) -> float:
    e = pd.Series(expected).astype("string").fillna("MISSING").value_counts(normalize=True)
    a = pd.Series(actual).astype("string").fillna("MISSING").value_counts(normalize=True)
    cats = e.index.union(a.index)
    e, a = e.reindex(cats, fill_value=0).clip(lower=eps), a.reindex(cats, fill_value=0).clip(lower=eps)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float, investigate: float = 0.10, action: float = 0.25) -> str:
    if np.isnan(value):
        return "no data"
    if value < investigate:
        return "stable"
    return "investigate" if value < action else "large shift"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str],
                 investigate=0.10, action=0.25) -> pd.DataFrame:
    rows = []
    for col in numeric:
        v = psi(reference[col], current[col])
        rows.append({"feature": col, "psi": v, "missing_ref": reference[col].isna().mean(),
                     "missing_now": current[col].isna().mean(), "status": status(v, investigate, action)})
    for col in categorical:
        v = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "psi": v, "missing_ref": 0.0, "missing_now": 0.0,
                     "status": status(v, investigate, action)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def quarterly_replay(reference: pd.DataFrame, later: pd.DataFrame, score_col: str, numeric: list[str],
                     categorical: list[str], investigate=0.10, action=0.25) -> pd.DataFrame:
    """For every later issue quarter: score PSI, the most shifted feature, and outcome-based checks."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for q, g in later.groupby(later["issue_date"].dt.to_period("Q")):
        rep = drift_report(reference, g, numeric, categorical, investigate, action)
        score_psi = psi(reference[score_col], g[score_col])
        rows.append({
            "quarter": str(q), "loans": len(g), "score_psi": score_psi,
            "score_status": status(score_psi, investigate, action),
            "most_shifted_feature": rep.iloc[0]["feature"], "its_psi": rep.iloc[0]["psi"],
            "features_to_investigate": int((rep["psi"] >= investigate).sum()),
            "mean_predicted_pd": g[score_col].mean(), "observed_bad_rate": g["bad"].mean(),
            "auc": roc_auc_score(g["bad"], g[score_col]) if g["bad"].nunique() == 2 else np.nan,
        })
    return pd.DataFrame(rows)


def vintage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bad rate by issue quarter: how each 'vintage' of loans ended up."""
    g = df.groupby(df["issue_date"].dt.to_period("Q"))["bad"].agg(["size", "mean"]).reset_index()
    g.columns = ["issue_quarter", "loans", "bad_rate"]
    g["issue_quarter"] = g["issue_quarter"].astype(str)
    return g
