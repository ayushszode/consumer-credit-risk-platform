"""Data-quality checks on the modelling population.

Checks report problems rather than silently fixing them, so anything odd is visible in the
results and can be discussed (which is what a model validator would ask for).
"""
from __future__ import annotations

import pandas as pd

from .features import MODEL_FEATURES

RANGES = {
    "loan_amnt": (500, 40_000), "int_rate": (4, 35), "fico_mid": (300, 850), "dti": (0, 100),
    "revol_util": (0, 150), "annual_inc": (1, 10_000_000), "credit_history_months": (0, 900),
}


def run_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, detail: str):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("loan ids are unique", df["id"].is_unique, f"{int(df['id'].duplicated().sum())} duplicates")
    add("target is 0/1 with no gaps", df["bad"].isin([0, 1]).all(), f"bad rate {df['bad'].mean():.2%}")
    add("issue dates present", df["issue_date"].notna().all(),
        f"{df['issue_date'].min():%Y-%m} to {df['issue_date'].max():%Y-%m}")
    for col, (lo, hi) in RANGES.items():
        if col in df:
            out = int(((df[col] < lo) | (df[col] > hi)).sum())
            add(f"{col} within [{lo:,}, {hi:,}]", out == 0, f"{out} loans outside the range (kept, flagged)")
    missing = df[MODEL_FEATURES].isna().mean().sort_values(ascending=False)
    worst = ", ".join(f"{c} {v:.1%}" for c, v in missing.head(6).items() if v > 0)
    add("missing values reviewed", True, worst or "no missing values in model features")
    return pd.DataFrame(checks)


def missing_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Share missing for each model feature by issue year (spots fields that appear/disappear over time)."""
    return (df.groupby(df["issue_date"].dt.year)[MODEL_FEATURES]
              .apply(lambda g: g.isna().mean()).T.round(3))
