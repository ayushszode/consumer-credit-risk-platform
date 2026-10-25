"""Tools for planning a lending-policy experiment (champion vs challenger).

Important distinction:
  * An OFFLINE comparison re-scores loans that were already funded under the old policy. It is
    cheap and useful, but it is not a causal experiment: we only see outcomes for loans that the
    old process approved.
  * A LIVE experiment randomly sends real applicants to each policy and waits for outcomes. For
    36-month loans, outcomes take years to mature, which is why early-warning metrics
    (e.g. 3- or 6-month arrears) are used as guardrails.

This module only supports planning (sample sizes) and offline comparison. Anything produced by
`simulate_random_split` is labelled as simulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def two_proportion_ztest(bad_a: int, n_a: int, bad_b: int, n_b: int) -> tuple[float, float]:
    p_a, p_b = bad_a / n_a, bad_b / n_b
    pooled = (bad_a + bad_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def required_sample_size(p_base: float, min_detectable_diff: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate loans per arm to detect a given absolute change in bad rate."""
    p_new = p_base + min_detectable_diff
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    var = p_base * (1 - p_base) + p_new * (1 - p_new)
    return int(np.ceil((z_a + z_b) ** 2 * var / min_detectable_diff ** 2))


def offline_policy_comparison(df: pd.DataFrame, champion_pd: str, champion_cut: float,
                              challenger_pd: str, challenger_cut: float) -> pd.DataFrame:
    """Same historical loans, two policies. Shows who each policy would have selected."""
    a = df[champion_pd] <= champion_cut
    b = df[challenger_pd] <= challenger_cut
    rows = []
    for name, sel in [("champion", a), ("challenger", b), ("both select", a & b),
                      ("only champion", a & ~b), ("only challenger", ~a & b)]:
        rows.append({"group": name, "loans": int(sel.sum()), "share_of_population": float(sel.mean()),
                     "observed_bad_rate": float(df.loc[sel, "bad"].mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows)


def simulate_random_split(df: pd.DataFrame, champion_select: np.ndarray, challenger_select: np.ndarray,
                          seed: int = 7, alpha: float = 0.05) -> dict:
    """SIMULATED: randomly assign historical loans to each arm and compare bad rates of selected loans."""
    rng = np.random.default_rng(seed)
    arm_b = rng.random(len(df)) < 0.5
    y = df["bad"].to_numpy()
    sel_a, sel_b = (~arm_b) & np.asarray(champion_select), arm_b & np.asarray(challenger_select)
    n_a, n_b, bad_a, bad_b = int(sel_a.sum()), int(sel_b.sum()), int(y[sel_a].sum()), int(y[sel_b].sum())
    z, p = two_proportion_ztest(bad_a, n_a, bad_b, n_b)
    return {"label": "SIMULATED on historical funded loans - not a live experiment",
            "champion_selected": n_a, "challenger_selected": n_b,
            "champion_bad_rate": bad_a / max(n_a, 1), "challenger_bad_rate": bad_b / max(n_b, 1),
            "z": z, "p_value": p, "significant_at_5pct": bool(p < alpha)}
