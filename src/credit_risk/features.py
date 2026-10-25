"""Which columns the models are allowed to use, and how derived features are built.

The most important rule in this project: a model may only use information a lender would have
at the moment of the application. Anything recorded after the loan was issued (payments,
recoveries, later credit pulls, hardship plans) would leak the outcome into the model.

`build_features` is used both when training and when scoring through the API/app, so the two
can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Applicant and credit-bureau information available at application time.
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "loan_to_income", "dti", "fico_mid", "emp_length_years",
    "credit_history_months", "inq_last_6mths", "mths_since_recent_inq", "delinq_2yrs",
    "mths_since_last_delinq", "pub_rec", "pub_rec_bankruptcies", "open_acc", "total_acc",
    "acc_open_past_24mths", "num_actv_rev_tl", "mort_acc", "revol_bal", "revol_util", "bc_util",
    "percent_bc_gt_75", "total_rev_hi_lim", "tot_cur_bal", "mo_sin_rcnt_tl",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Produced by LendingClub's own credit assessment and pricing. Used only in a separate
# benchmark model, never in the applicant-only models.
LENDER_FIELDS = ["grade", "sub_grade", "int_rate", "installment"]

# Needed for building the population, splits and segments, but not as model inputs.
KEY_FIELDS = ["id", "issue_date", "loan_status", "term_months", "policy_code", "application_type",
              "addr_state"]

# Only known after the loan finishes. Used to estimate loss given default (LGD) and exposure,
# never as model inputs.
LOSS_FIELDS = ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]

# Raw columns read from the source file.
RAW_NUMERIC = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "policy_code", "mort_acc",
    "pub_rec_bankruptcies", "bc_util", "acc_open_past_24mths", "num_actv_rev_tl", "tot_cur_bal",
    "total_rev_hi_lim", "percent_bc_gt_75", "mo_sin_rcnt_tl", "mths_since_recent_inq",
    "total_rec_prncp", "recoveries", "collection_recovery_fee",
]
RAW_TEXT = ["id", "term", "grade", "sub_grade", "emp_length", "home_ownership", "verification_status",
            "issue_d", "loan_status", "purpose", "addr_state", "earliest_cr_line", "application_type"]
RAW_COLUMNS = RAW_TEXT + RAW_NUMERIC

# Fields that must never appear in a model's feature list.
LEAKAGE_FIELDS = {
    "loan_status", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "out_prncp", "out_prncp_inv", "hardship_flag", "debt_settlement_flag",
    "pymnt_plan", "funded_amnt_inv", "settlement_status", "settlement_amount",
}

FEATURE_LABELS = {
    "loan_amnt": "Loan amount", "annual_inc": "Annual income", "loan_to_income": "Loan-to-income ratio",
    "dti": "Debt-to-income ratio (%)", "fico_mid": "Credit score (FICO)",
    "emp_length_years": "Years in current job", "credit_history_months": "Length of credit history (months)",
    "inq_last_6mths": "Credit enquiries in last 6 months", "mths_since_recent_inq": "Months since last enquiry",
    "delinq_2yrs": "Late payments in last 2 years", "mths_since_last_delinq": "Months since last late payment",
    "pub_rec": "Public records", "pub_rec_bankruptcies": "Bankruptcies", "open_acc": "Open credit lines",
    "total_acc": "Total credit lines", "acc_open_past_24mths": "Accounts opened in last 24 months",
    "num_actv_rev_tl": "Active revolving accounts", "mort_acc": "Mortgage accounts",
    "revol_bal": "Revolving balance", "revol_util": "Revolving credit used (%)",
    "bc_util": "Bank card credit used (%)", "percent_bc_gt_75": "Bank cards over 75% used (%)",
    "total_rev_hi_lim": "Total revolving credit limit", "tot_cur_bal": "Total current balance",
    "mo_sin_rcnt_tl": "Months since newest account", "home_ownership": "Home ownership",
    "verification_status": "Income verification", "purpose": "Loan purpose",
}


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole calendar months from start to end (matches DuckDB's date_diff('month', ...))."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features. Works on the cleaned table or on a single API applicant."""
    out = df.copy()
    if "fico_mid" not in out or out["fico_mid"].isna().all():
        out["fico_mid"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    if "credit_history_months" not in out or out["credit_history_months"].isna().all():
        out["credit_history_months"] = months_between(out["earliest_cr_date"], out["issue_date"])
    income = out["annual_inc"].where(out["annual_inc"] > 0)
    out["loan_to_income"] = out["loan_amnt"] / income
    # plain float64 with NaN for missing (scikit-learn doesn't accept pandas' nullable types)
    for col in NUMERIC_FEATURES + [c for c in ("int_rate", "installment") if c in out]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64") if col in out else np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in out:
            out[col] = None
        out[col] = out[col].astype("string").str.strip().str.upper().fillna("MISSING")
    return out


def assert_no_leakage(features: list[str]) -> None:
    leaked = set(features) & (LEAKAGE_FIELDS | set(LOSS_FIELDS))
    if leaked:
        raise ValueError(f"Post-origination fields used as features: {sorted(leaked)}")


def feature_register(header: list[str] | None = None) -> pd.DataFrame:
    """One row per source column: is it known at application time, and what we do with it.

    If the real file's header is given, any column not covered by the rules below is marked
    'needs review' instead of being silently ignored.
    """
    rows = []
    derived = {"fico_range_low": "fico_mid", "fico_range_high": "fico_mid",
               "earliest_cr_line": "credit_history_months", "emp_length": "emp_length_years",
               "issue_d": "issue_date / credit_history_months", "term": "term_months"}
    columns = header or sorted(set(RAW_COLUMNS) | LEAKAGE_FIELDS)
    for col in columns:
        if col in MODEL_FEATURES or col in derived:
            row = (True, "use", f"Application-time information{'; used to build ' + derived[col] if col in derived else ''}")
        elif col in LENDER_FIELDS:
            row = (True, "benchmark only", "Set by the lender's own grading/pricing, not by the applicant")
        elif col in LEAKAGE_FIELDS or col in LOSS_FIELDS:
            row = (False, "exclude (leakage)", "Recorded after the loan was issued")
        elif col in {"id", "member_id", "url"}:
            row = (True, "exclude (identifier)", "Identifier, no predictive meaning")
        elif col in {"desc", "title", "emp_title", "zip_code"}:
            row = (True, "exclude (free text / location detail)", "Free text or partial postcode; privacy and proxy risk")
        elif col in {"addr_state"}:
            row = (True, "segments only", "Used for monitoring by region, not as a model input")
        elif col in {"policy_code", "application_type", "loan_status"}:
            row = (col != "loan_status", "population rule", "Used to define the eligible population or the target")
        elif col.startswith(("sec_app_", "hardship_", "settlement_", "debt_settlement", "orig_projected",
                             "deferral_", "payment_plan")) or col.endswith("_joint"):
            row = (False, "exclude", "Joint-application or post-origination field")
        else:
            row = (None, "needs review", "Not yet classified; excluded until reviewed")
        rows.append({"field": col, "available_at_application": row[0], "decision": row[1], "reason": row[2]})
    return pd.DataFrame(rows)
