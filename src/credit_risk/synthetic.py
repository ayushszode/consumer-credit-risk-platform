"""Fake loans in the same format as the LendingClub file.

Used for the tests, the GitHub Actions checks and the public demo, so no real borrower records
ever need to be committed or shown. The relationships are invented: these numbers say nothing
about real credit risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
            "small_business", "car", "medical"]
HOMES = ["MORTGAGE", "RENT", "OWN"]
VERIF = ["Not Verified", "Source Verified", "Verified"]
STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH"]


def make_raw_loans(n: int = 20_000, seed: int = 0, start: str = "2011-01-01", end: str = "2016-06-30",
                   add_footer: bool = True) -> pd.DataFrame:
    """Return a DataFrame of text columns that looks like the raw accepted-loans file."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="MS")
    issue = pd.to_datetime(rng.choice(months, n))
    t = (issue.year - 2011) + issue.month / 12  # time trend for a little drift

    fico_low = np.clip(rng.normal(695, 30, n).round() // 5 * 5, 660, 845)
    income = np.exp(rng.normal(11.0, 0.5, n)).round(-2)
    loan = np.clip(rng.lognormal(9.4, 0.6, n).round(-2), 1000, 40000)
    dti = np.clip(rng.normal(18 + 0.4 * t, 8, n), 0, 45).round(2)
    inq = rng.poisson(0.8, n)
    revol_util = np.clip(rng.normal(52, 24, n), 0, 140).round(1)
    emp = rng.integers(0, 11, n)
    history = np.clip(rng.normal(190, 85, n), 24, 600).astype(int)
    delinq = rng.poisson(0.3, n)
    purpose = rng.choice(PURPOSES, n, p=[.55, .2, .07, .06, .04, .03, .03, .02])
    home = rng.choice(HOMES, n, p=[.48, .41, .11])
    verif = rng.choice(VERIF, n, p=[.33, .37, .30])
    term = rng.choice([36, 60], n, p=[.75, .25])

    logit = (-2.1 - 0.018 * (fico_low - 695) + 0.035 * (dti - 18) + 0.22 * inq + 0.006 * (revol_util - 52)
             - 0.25 * np.log(income / 60000) + 0.25 * np.log(loan / 12000) - 0.02 * emp
             - 0.0015 * (history - 190) + 0.15 * delinq + 0.45 * (purpose == "small_business")
             + 0.2 * (home == "RENT") + 0.15 * (verif == "Verified") + 0.5 * (term == 60) + 0.05 * t)
    p_bad = 1 / (1 + np.exp(-logit))
    bad = rng.random(n) < p_bad

    grade_idx = np.clip(((p_bad - 0.02) / 0.06).astype(int) + rng.integers(-1, 2, n), 0, 6)
    grades = np.array(list("ABCDEFG"))[grade_idx]
    sub = [f"{g}{rng.integers(1, 6)}" for g in grades]
    rate = (6 + grade_idx * 3.5 + rng.normal(0, 0.8, n)).round(2)

    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    recent = issue > pd.Timestamp("2015-10-01")
    unresolved = recent & (rng.random(n) < 0.15)
    status[unresolved] = rng.choice(["Current", "Late (31-120 days)", "In Grace Period"], unresolved.sum())
    status[(~unresolved) & bad & (rng.random(n) < 0.03)] = "Default"

    earliest = issue - pd.to_timedelta(history * 30.44, unit="D")
    total_rec_prncp = np.where(bad, loan * rng.uniform(0.1, 0.7, n), loan).round(2)
    recoveries = np.where(bad, (loan - total_rec_prncp) * rng.uniform(0, 0.2, n), 0).round(2)

    df = pd.DataFrame({
        "id": (np.arange(n) + 1_000_000).astype(str),
        "loan_amnt": loan, "funded_amnt": loan, "term": [f" {x} months" for x in term],
        "int_rate": rate, "installment": (loan * (rate / 1200) / (1 - (1 + rate / 1200) ** -term)).round(2),
        "grade": grades, "sub_grade": sub,
        "emp_length": np.where(emp == 0, "< 1 year", np.where(emp == 10, "10+ years",
                               [f"{e} year" + ("s" if e > 1 else "") for e in emp])),
        "home_ownership": home, "annual_inc": income, "verification_status": verif,
        "issue_d": issue.strftime("%b-%Y"), "loan_status": status, "purpose": purpose,
        "addr_state": rng.choice(STATES, n), "dti": dti, "delinq_2yrs": delinq,
        "earliest_cr_line": earliest.strftime("%b-%Y"),
        "fico_range_low": fico_low, "fico_range_high": fico_low + 4, "inq_last_6mths": inq,
        "mths_since_last_delinq": np.where(delinq > 0, rng.integers(1, 24, n), np.nan),
        "open_acc": np.clip(rng.poisson(11, n), 1, None), "pub_rec": rng.poisson(0.15, n),
        "revol_bal": (rng.lognormal(9.3, 0.9, n)).round(), "revol_util": revol_util,
        "total_acc": np.clip(rng.poisson(24, n), 2, None), "policy_code": 1,
        "application_type": np.where(rng.random(n) < 0.02, "Joint App", "Individual"),
        "mort_acc": np.where(home == "MORTGAGE", rng.poisson(2, n), rng.poisson(0.3, n)),
        "pub_rec_bankruptcies": rng.poisson(0.1, n), "bc_util": np.clip(revol_util + rng.normal(5, 10, n), 0, 150).round(1),
        "acc_open_past_24mths": rng.poisson(4 + inq, n), "num_actv_rev_tl": rng.poisson(5, n),
        "tot_cur_bal": (rng.lognormal(11, 1.1, n)).round(), "total_rev_hi_lim": (rng.lognormal(10.2, 0.7, n)).round(),
        "percent_bc_gt_75": np.clip(rng.normal(40, 30, n), 0, 100).round(1),
        "mo_sin_rcnt_tl": rng.integers(0, 40, n), "mths_since_recent_inq": np.where(inq > 0, rng.integers(0, 6, n), rng.integers(6, 24, n)),
        "total_rec_prncp": total_rec_prncp, "recoveries": recoveries,
        "collection_recovery_fee": (recoveries * 0.18).round(2),
        "total_pymnt": (total_rec_prncp * 1.1).round(2),  # leakage column, present in the real file
    })
    # a few missing values, like the real file
    for col, share in [("dti", .002), ("revol_util", .001), ("bc_util", .01), ("mths_since_recent_inq", .1)]:
        df.loc[rng.random(n) < share, col] = np.nan
    df = df.astype(str).replace({"nan": ""})
    if add_footer:
        footer = pd.DataFrame([{"id": "Total amount funded in policy code 1: 123456"},
                               {"id": "Total amount funded in policy code 2: 7890"}])
        df = pd.concat([df, footer], ignore_index=True).fillna("")
    return df


def write_raw_file(path, n: int = 20_000, seed: int = 0) -> None:
    make_raw_loans(n=n, seed=seed).to_csv(path, index=False, compression="gzip")


def synthetic_applicant() -> dict:
    """One made-up applicant for the API and app demos."""
    return {
        "loan_amnt": 12000, "annual_inc": 38000, "dti": 31.0, "fico_range_low": 675, "fico_range_high": 679,
        "emp_length_years": 3, "earliest_cr_line": "2016-04", "inq_last_6mths": 2, "mths_since_recent_inq": 2,
        "delinq_2yrs": 0, "mths_since_last_delinq": None, "pub_rec": 0, "pub_rec_bankruptcies": 0,
        "open_acc": 9, "total_acc": 18, "acc_open_past_24mths": 6, "num_actv_rev_tl": 6, "mort_acc": 0,
        "revol_bal": 9800, "revol_util": 78.0, "bc_util": 82.0, "percent_bc_gt_75": 60.0,
        "total_rev_hi_lim": 14000, "tot_cur_bal": 21000, "mo_sin_rcnt_tl": 3,
        "home_ownership": "RENT", "verification_status": "Source Verified", "purpose": "debt_consolidation",
    }
