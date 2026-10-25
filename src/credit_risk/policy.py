"""Turning a predicted PD into a lending decision.

The policy is deliberately simple and lives in configs/base.yaml:
  PD <= approve_max_pd           -> approve
  approve_max_pd < PD < decline  -> refer to a person (manual review)
  PD >= decline_min_pd           -> decline

Everything here is a historical, offline analysis on loans LendingClub actually funded. It shows
the trade-off between approving more loans and taking on more risk. It is NOT evidence of what
would happen if the policy were used on all applicants, because we never see how rejected
applicants would have behaved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_band(pd_values, bands: dict[str, float]) -> np.ndarray:
    pd_values = np.asarray(pd_values, dtype=float)
    labels, limits = list(bands.keys()), list(bands.values())
    idx = np.searchsorted(limits, pd_values, side="left")
    return np.array(labels)[np.minimum(idx, len(labels) - 1)]


def decide(pd_value: float, cfg_policy: dict) -> str:
    if pd_value <= cfg_policy["approve_max_pd"]:
        return "approve"
    if pd_value >= cfg_policy["decline_min_pd"]:
        return "decline"
    return "refer"


def band_table(df: pd.DataFrame, pd_col: str, bands: dict[str, float]) -> pd.DataFrame:
    d = df.assign(band=risk_band(df[pd_col], bands))
    out = (d.groupby("band")
             .agg(loans=("bad", "size"), bad_loans=("bad", "sum"), bad_rate=("bad", "mean"),
                  mean_predicted_pd=(pd_col, "mean"))
             .reindex(list(bands.keys())).dropna(how="all").reset_index())
    out["share_of_loans"] = out["loans"] / out["loans"].sum()
    return out


def threshold_curve(df: pd.DataFrame, pd_col: str, grid=None) -> pd.DataFrame:
    """If we had only funded loans with PD <= cut-off, what share would we have kept and how bad were they?"""
    grid = np.round(np.arange(0.04, 0.41, 0.01), 2) if grid is None else grid
    total = len(df)
    rows = []
    for c in grid:
        kept = df[df[pd_col] <= c]
        rows.append({"cutoff_pd": float(c), "selected_share": len(kept) / total,
                     "observed_bad_rate": kept["bad"].mean() if len(kept) else np.nan,
                     "bad_loans_avoided_share": 1 - kept["bad"].sum() / max(df["bad"].sum(), 1)})
    return pd.DataFrame(rows)


def outcome_summary(df: pd.DataFrame, pd_col: str, cfg_policy: dict) -> pd.DataFrame:
    d = df.assign(outcome=[decide(p, cfg_policy) for p in df[pd_col]])
    out = d.groupby("outcome").agg(loans=("bad", "size"), observed_bad_rate=("bad", "mean")).reset_index()
    out["share"] = out["loans"] / out["loans"].sum()
    return out.set_index("outcome").reindex(["approve", "refer", "decline"]).dropna(how="all").reset_index()
