"""Scoring one applicant: the single code path used by the API, the Streamlit app and the tests.

Input: a dict of application-time information (see api/schemas in api/main.py).
Output: PD, risk band, policy outcome, reason codes, versions and a plain-English explanation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .explain import build_facts, llm_explanation
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, build_features
from .policy import decide, risk_band

DEFAULT_BUNDLE = PROJECT_ROOT / "artifacts" / "model_bundle.joblib"


def load_bundle(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_BUNDLE
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m credit_risk.pipeline` first.")
    return joblib.load(path)


def applicant_frame(applicant: dict) -> pd.DataFrame:
    """Turn the API/app input into the same columns the models were trained on."""
    a = dict(applicant)
    app_date = pd.Timestamp(a.pop("application_date", None) or date.today())
    row = {c: a.get(c) for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    row["fico_range_low"] = a.get("fico_range_low")
    row["fico_range_high"] = a.get("fico_range_high", a.get("fico_range_low"))
    row["issue_date"] = app_date
    row["earliest_cr_date"] = pd.to_datetime(a.get("earliest_cr_line"), format="%Y-%m", errors="coerce")
    row["fico_mid"] = np.nan
    row["credit_history_months"] = np.nan
    df = pd.DataFrame([row])
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fico_range_low"] = pd.to_numeric(df["fico_range_low"], errors="coerce")
    df["fico_range_high"] = pd.to_numeric(df["fico_range_high"], errors="coerce")
    return build_features(df)


def predict_pd(bundle: dict, X: pd.DataFrame, which: str | None = None) -> np.ndarray:
    which = which or bundle["champion"]
    if which == "scorecard":
        return bundle["scorecard"].predict_proba(X)[:, 1]
    return bundle["gbm"].predict_proba(X[MODEL_FEATURES])[:, 1]


def gbm_reason_codes(bundle: dict, X: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """For the boosting model: features where swapping in a typical value lowers PD the most."""
    base = predict_pd(bundle, X, "gbm")[0]
    rows = []
    for f, typical in bundle["typical_values"].items():
        Xc = X.copy()
        Xc[f] = typical
        drop = base - predict_pd(bundle, Xc, "gbm")[0]
        if drop > 0:
            v = X.iloc[0][f]
            rows.append({"feature": f, "value": None if pd.isna(v) else (v if isinstance(v, str) else float(v)),
                         "pd_reduction_if_typical": float(drop)})
    rows.sort(key=lambda r: r["pd_reduction_if_typical"], reverse=True)
    return rows[:top_n]


def score_applicant(bundle: dict, applicant: dict, call_llm=None) -> dict:
    X = applicant_frame(applicant)
    pd_value = float(predict_pd(bundle, X)[0])
    if bundle["champion"] == "scorecard":
        reasons = bundle["scorecard"].reason_codes(X)[0]
        points = float(bundle["scorecard"].score(X).iloc[0])
    else:
        reasons = gbm_reason_codes(bundle, X)
        points = float(bundle["scorecard"].score(X).iloc[0])  # shown for reference only
    policy = bundle["policy"]
    decision = {
        "probability_of_default": round(pd_value, 4),
        "target_definition": bundle["target_definition"],
        "risk_band": str(risk_band([pd_value], policy["bands"])[0]),
        "policy_outcome": decide(pd_value, policy),
        "scorecard_points": points,
        "reason_codes": reasons,
        "model": bundle["champion"],
        "model_version": bundle["model_version"],
        "policy_version": policy["version"],
    }
    decision["explanation"] = llm_explanation(build_facts(decision), call_llm)
    return decision
