"""Charts saved to reports/figures. Kept plain on purpose: one message per chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

COLOURS = {"scorecard": "#185FA5", "gbm": "#D85A30", "benchmark": "#888780"}
NAMES = {"scorecard": "Scorecard (logistic, WoE)", "gbm": "Gradient boosting",
         "benchmark": "Benchmark incl. lender grade"}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def _save(fig, out: Path, name: str, note: str | None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color="#A32D2D")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def roc_chart(y, scores: dict[str, np.ndarray], out: Path, note=None):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for k, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=COLOURS[k], label=NAMES[k], linewidth=1.6)
    ax.plot([0, 1], [0, 1], color="#B4B2A9", linestyle="--", linewidth=1)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "ROC curve, test year", "Share of good loans flagged", "Share of bad loans caught")
    _save(fig, out, "roc_test.png", note)


def calibration_chart(tables: dict[str, pd.DataFrame], out: Path, note=None):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    top = 0
    for k, t in tables.items():
        ax.plot(t["predicted_pd"], t["actual_bad_rate"], marker="o", markersize=4, color=COLOURS[k], label=NAMES[k])
        top = max(top, t["predicted_pd"].max(), t["actual_bad_rate"].max())
    ax.plot([0, top], [0, top], color="#B4B2A9", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Calibration, test year (10 groups)", "Average predicted PD", "Actual bad rate")
    _save(fig, out, "calibration_test.png", note)


def tradeoff_chart(curve: pd.DataFrame, out: Path, note=None):
    curve = curve[curve["selected_share"] >= 0.02]  # very small groups are just noise
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(curve["selected_share"] * 100, curve["observed_bad_rate"] * 100, color=COLOURS["scorecard"], linewidth=1.8)
    for c in (0.08, 0.12, 0.16, 0.20):
        row = curve.iloc[(curve["cutoff_pd"] - c).abs().argmin()]
        ax.annotate(f"PD ≤ {c:.0%}", (row["selected_share"] * 100, row["observed_bad_rate"] * 100), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
        ax.plot(row["selected_share"] * 100, row["observed_bad_rate"] * 100, "o", color=COLOURS["gbm"], markersize=4)
    _style(ax, "Approval vs bad rate on historical funded loans", "Loans selected (%)", "Bad rate of selected loans (%)")
    _save(fig, out, "policy_tradeoff.png", note)


def vintage_chart(v: pd.DataFrame, out: Path, note=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(v["issue_quarter"], v["bad_rate"] * 100, color="#85B7EB")
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(v["issue_quarter"], rotation=60, fontsize=7)
    _style(ax, "Bad rate by issue quarter (vintage)", "", "Bad rate (%)")
    _save(fig, out, "vintages.png", note)


def psi_chart(replay: pd.DataFrame, investigate: float, action: float, out: Path, note=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(replay["quarter"], replay["score_psi"], marker="o", color=COLOURS["scorecard"], label="Score PSI")
    ax.plot(replay["quarter"], replay["its_psi"], marker="s", color=COLOURS["gbm"], label="Most shifted feature")
    ax.axhline(investigate, color="#BA7517", linestyle="--", linewidth=1, label=f"Investigate ({investigate})")
    ax.axhline(action, color="#A32D2D", linestyle="--", linewidth=1, label=f"Large shift ({action})")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "Monitoring replay against the training years", "Issue quarter", "PSI")
    _save(fig, out, "monitoring_psi.png", note)


def iv_chart(iv: pd.DataFrame, out: Path, note=None):
    top = iv.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    colours = [COLOURS["scorecard"] if s == "selected" else "#D3D1C7" for s in top["status"]]
    ax.barh(top["feature"], top["iv"], color=colours)
    _style(ax, "Information Value (blue = used in scorecard)", "IV", "")
    _save(fig, out, "information_value.png", note)
