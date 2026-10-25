"""Scoring API.

Run locally:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try it.

Every response carries the model and policy versions, so any decision can be traced back to the
exact model that made it.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.service import load_bundle, score_applicant

API_VERSION = "1.0"
app = FastAPI(title="Consumer credit risk scoring API", version=API_VERSION,
              description="Educational project. Trained on historical US loans; not for real lending decisions.")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle(os.getenv("MODEL_BUNDLE"))
    return _bundle


class Applicant(BaseModel):
    """Information known at the time of the application. Ranges reject obviously wrong input."""
    loan_amnt: float = Field(..., ge=500, le=40000, description="Requested amount")
    annual_inc: float = Field(..., gt=0, le=10_000_000)
    dti: float = Field(..., ge=0, le=100, description="Monthly debt payments / monthly income, in %")
    fico_range_low: int = Field(..., ge=300, le=850)
    fico_range_high: Optional[int] = Field(None, ge=300, le=850)
    emp_length_years: Optional[float] = Field(None, ge=0, le=10)
    earliest_cr_line: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="First credit account, YYYY-MM")
    application_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    inq_last_6mths: int = Field(0, ge=0, le=50)
    mths_since_recent_inq: Optional[float] = Field(None, ge=0)
    delinq_2yrs: int = Field(0, ge=0, le=50)
    mths_since_last_delinq: Optional[float] = Field(None, ge=0)
    pub_rec: int = Field(0, ge=0)
    pub_rec_bankruptcies: int = Field(0, ge=0)
    open_acc: int = Field(..., ge=0)
    total_acc: int = Field(..., ge=0)
    acc_open_past_24mths: Optional[int] = Field(None, ge=0)
    num_actv_rev_tl: Optional[int] = Field(None, ge=0)
    mort_acc: Optional[int] = Field(None, ge=0)
    revol_bal: float = Field(..., ge=0)
    revol_util: Optional[float] = Field(None, ge=0, le=200)
    bc_util: Optional[float] = Field(None, ge=0, le=200)
    percent_bc_gt_75: Optional[float] = Field(None, ge=0, le=100)
    total_rev_hi_lim: Optional[float] = Field(None, ge=0)
    tot_cur_bal: Optional[float] = Field(None, ge=0)
    mo_sin_rcnt_tl: Optional[float] = Field(None, ge=0)
    home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    verification_status: Literal["Not Verified", "Source Verified", "Verified"]
    purpose: str = Field(..., min_length=2, max_length=40)


class Explanation(BaseModel):
    text: str
    source: Literal["template", "llm"]
    problems: list[str]


class Decision(BaseModel):
    api_version: str
    probability_of_default: float
    target_definition: str
    risk_band: str
    policy_outcome: Literal["approve", "refer", "decline"]
    scorecard_points: float
    reason_codes: list[dict]
    model: str
    model_version: str
    policy_version: str
    explanation: Explanation


@app.get("/health")
def health():
    try:
        b = bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "ok", "model_version": b["model_version"], "policy_version": b["policy"]["version"],
            "trained_on": b["trained_on"], "synthetic_model": b.get("synthetic", False)}


@app.post("/score", response_model=Decision)
def score(applicant: Applicant):
    try:
        b = bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if applicant.fico_range_high is not None and applicant.fico_range_high < applicant.fico_range_low:
        raise HTTPException(status_code=422, detail="fico_range_high must be at least fico_range_low")
    applied = (applicant.application_date or date.today().isoformat())[:7]
    if applicant.earliest_cr_line > applied:
        raise HTTPException(status_code=422, detail="earliest_cr_line cannot be after the application date")
    result = score_applicant(b, applicant.model_dump())
    return {"api_version": API_VERSION, **result}
