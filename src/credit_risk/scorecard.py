"""A classic credit scorecard: binning -> Weight of Evidence -> logistic regression -> points.

How it works, in plain terms:
  1. Each feature is split into a handful of bins (e.g. FICO 660-679, 680-699, ...). Missing values
     get their own bin, because "no record" often means something.
  2. Each bin gets a Weight of Evidence (WoE) = ln(% of all good loans in the bin / % of all bad loans
     in the bin). Positive WoE = safer than average, negative = riskier.
  3. Information Value (IV) adds up how different good and bad loans look across the bins. Features
     with IV below `min_iv` are dropped; of two highly correlated features we keep the stronger one.
  4. A logistic regression is fitted on the WoE values.
  5. The result is scaled to points: `base_score` points at `base_odds` good:bad, and every `pdo`
     points doubles the odds. Each applicant's score is the sum of the points for their bins.

Reason codes come straight from the points: the characteristics where an applicant scored furthest
below the best possible points are the main reasons their risk is higher.

All bins and WoE values are learned on the training data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 0.5  # small count added to every bin so WoE is never log(0)


@dataclass
class FeatureBins:
    name: str
    kind: str                       # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)       # numeric: inner cut points
    categories: dict[str, str] = field(default_factory=dict)  # categorical: value -> bin label
    table: pd.DataFrame | None = None                        # bin, n, bad, woe, iv
    iv: float = 0.0

    def assign(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            values = pd.to_numeric(x, errors="coerce")
            idx = np.searchsorted(self.edges, values.to_numpy(dtype=float), side="right")
            labels = pd.Series([self._label(i) for i in idx], index=x.index, dtype="object")
            return labels.where(values.notna(), "Missing")
        mapped = x.astype("string").fillna("MISSING").map(self.categories)
        return mapped.fillna("OTHER").astype("object")

    def _label(self, i: int) -> str:
        lo = "-inf" if i == 0 else _fmt(self.edges[i - 1])
        hi = "inf" if i == len(self.edges) else _fmt(self.edges[i])
        return f"[{lo}, {hi})"

    def woe(self, x: pd.Series) -> pd.Series:
        lookup = self.table.set_index("bin")["woe"]
        return self.assign(x).map(lookup).fillna(0.0).astype(float)  # unseen bin -> neutral


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"bin": bins.to_numpy(), "bad": y.to_numpy()}).groupby("bin", sort=False)["bad"].agg(["size", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    total_good, total_bad = t["good"].sum(), t["bad"].sum()
    dist_good = (t["good"] + EPS) / (total_good + EPS * len(t))
    dist_bad = (t["bad"] + EPS) / (total_bad + EPS * len(t))
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t.reset_index()


def _numeric_edges(x: pd.Series, y: pd.Series, n_bins: int, min_share: float) -> list[float]:
    """Quantile cut points, then merge neighbouring bins that are too small or have equal bad rates."""
    values = pd.to_numeric(x, errors="coerce").dropna()
    if values.nunique() <= 1:
        return []
    qs = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = list(qs)
    target = y.loc[values.index]
    min_n = max(int(min_share * len(x)), 1)

    def stats(e):
        idx = np.searchsorted(e, values.to_numpy(), side="right")
        g = pd.DataFrame({"b": idx, "y": target.to_numpy()}).groupby("b")["y"].agg(["size", "mean"])
        return g.reindex(range(len(e) + 1)).fillna({"size": 0, "mean": 0.0})

    changed = True
    while changed and edges:
        changed = False
        g = stats(edges)
        small = g.index[g["size"] < min_n].tolist()
        if small:
            b = small[0]
            del edges[min(b, len(edges) - 1)]  # merge with the next bin (or the previous one at the end)
            changed = True
            continue
        diffs = np.abs(np.diff(g["mean"].to_numpy()))
        if len(diffs) and diffs.min() < 0.002:   # neighbours with almost the same bad rate add nothing
            del edges[int(diffs.argmin())]
            changed = True
    return edges


def fit_bins(x: pd.Series, y: pd.Series, kind: str, n_bins: int = 8, min_share: float = 0.05) -> FeatureBins:
    fb = FeatureBins(name=x.name, kind=kind)
    if kind == "numeric":
        fb.edges = _numeric_edges(x, y, n_bins, min_share)
    else:
        values = x.astype("string").fillna("MISSING")
        share = values.value_counts(normalize=True)
        fb.categories = {v: (v if s >= min_share else "OTHER") for v, s in share.items()}
    table = _woe_table(fb.assign(x), y)
    if kind == "numeric":
        order = {fb._label(i): i for i in range(len(fb.edges) + 1)}
        order["Missing"] = len(order)
        table = table.sort_values("bin", key=lambda s: s.map(order)).reset_index(drop=True)
    else:
        table = table.sort_values("bad_rate").reset_index(drop=True)
    fb.table = table
    fb.iv = float(fb.table["iv"].sum())
    return fb


class Scorecard:
    """Fit with a DataFrame of raw features and a 0/1 target (1 = bad)."""

    def __init__(self, numeric: list[str], categorical: list[str], n_bins=8, min_bin_share=0.05,
                 min_iv=0.02, max_corr=0.70, C=1.0, base_score=600, base_odds=20, pdo=20):
        self.numeric, self.categorical = numeric, categorical
        self.n_bins, self.min_bin_share, self.min_iv, self.max_corr, self.C = n_bins, min_bin_share, min_iv, max_corr, C
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo
        self.bins: dict[str, FeatureBins] = {}
        self.selected: list[str] = []
        self.dropped: dict[str, str] = {}

    # ----- fitting -----
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        y = pd.Series(np.asarray(y), index=X.index)
        for col in self.numeric:
            self.bins[col] = fit_bins(X[col], y, "numeric", self.n_bins, self.min_bin_share)
        for col in self.categorical:
            self.bins[col] = fit_bins(X[col], y, "categorical", self.n_bins, self.min_bin_share)

        candidates = sorted((c for c in self.bins if self.bins[c].iv >= self.min_iv),
                            key=lambda c: self.bins[c].iv, reverse=True)
        for c in self.bins:
            if c not in candidates:
                self.dropped[c] = f"IV {self.bins[c].iv:.4g} below {self.min_iv}"
        W = self._woe_matrix(X, candidates)
        corr = W.corr().abs()
        for c in candidates:  # strongest first
            clash = [s for s in self.selected if corr.loc[c, s] > self.max_corr]
            if clash:
                self.dropped[c] = f"correlation {corr.loc[c, clash[0]]:.2f} with {clash[0]}"
            else:
                self.selected.append(c)

        self.model = LogisticRegression(C=self.C, max_iter=2000)
        self.model.fit(self._woe_matrix(X, self.selected), y)
        # Points scaling
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        self._build_points()
        return self

    def _woe_matrix(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({c: self.bins[c].woe(X[c]) for c in cols}, index=X.index)

    def _build_points(self) -> None:
        coefs = dict(zip(self.selected, self.model.coef_[0]))
        b0, n = self.model.intercept_[0], len(self.selected)
        rows = []
        for c in self.selected:
            t = self.bins[c].table.copy()
            # ln(odds of good) = -(b0 + sum b_i * woe_i), shared out equally across characteristics
            t["points"] = (-(coefs[c] * t["woe"] + b0 / n) * self.factor + self.offset / n).round(0)
            t["feature"] = c
            rows.append(t)
        self.points_table = pd.concat(rows, ignore_index=True)
        self._max_points = self.points_table.groupby("feature")["points"].max().to_dict()

    # ----- using the scorecard -----
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(self._woe_matrix(X, self.selected))[:, 1]
        return np.column_stack([1 - p, p])

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points per characteristic for each applicant (columns = selected features)."""
        out = {}
        for c in self.selected:
            lookup = self.points_table[self.points_table["feature"] == c].set_index("bin")["points"]
            neutral = float(np.round(self.offset / len(self.selected)))
            out[c] = self.bins[c].assign(X[c]).map(lookup).fillna(neutral).astype(float)
        return pd.DataFrame(out, index=X.index)

    def score(self, X: pd.DataFrame) -> pd.Series:
        return self.points(X).sum(axis=1)

    def reason_codes(self, X: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
        """For each applicant: the characteristics with the biggest gap to the best possible points."""
        pts = self.points(X)
        result = []
        for i in range(len(pts)):
            gaps = []
            for c in self.selected:
                gap = self._max_points[c] - pts.iloc[i][c]
                if gap > 0:
                    gaps.append({"feature": c, "value": _clean(X.iloc[i][c]), "bin": self.bins[c].assign(X.iloc[[i]][c]).iloc[0],
                                 "points": float(pts.iloc[i][c]), "points_below_best": float(gap)})
            gaps.sort(key=lambda r: r["points_below_best"], reverse=True)
            result.append(gaps[:top_n])
        return result

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "iv": b.iv, "bins": len(b.table),
                 "status": "selected" if c in self.selected else f"dropped: {self.dropped.get(c, '')}"}
                for c, b in self.bins.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return None if v is pd.NA else v


def iv_strength(iv: float) -> str:
    """Common rule of thumb for reading IV values."""
    if iv < 0.02:
        return "not useful"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspiciously strong - check for leakage"
