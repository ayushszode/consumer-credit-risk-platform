import os

from credit_risk.explain import build_facts, check_llm_output, llm_explanation, template_explanation
from credit_risk.service import score_applicant
from credit_risk.synthetic import synthetic_applicant


def test_score_applicant_output(bundle):
    r = score_applicant(bundle, synthetic_applicant())
    assert 0 <= r["probability_of_default"] <= 1
    assert r["policy_outcome"] in {"approve", "refer", "decline"}
    assert r["risk_band"] in bundle["policy"]["bands"]
    assert r["model_version"] == bundle["model_version"]
    assert r["explanation"]["source"] == "template"


def test_missing_optional_fields_still_score(bundle):
    a = synthetic_applicant()
    for k in ["bc_util", "mths_since_recent_inq", "mort_acc", "tot_cur_bal"]:
        a[k] = None
    assert 0 <= score_applicant(bundle, a)["probability_of_default"] <= 1


def test_safer_applicant_gets_lower_pd(bundle):
    risky = synthetic_applicant()
    safe = {**risky, "fico_range_low": 780, "fico_range_high": 784, "dti": 8, "revol_util": 15, "bc_util": 12,
            "inq_last_6mths": 0}
    assert score_applicant(bundle, safe)["probability_of_default"] < score_applicant(bundle, risky)["probability_of_default"]


def _facts(bundle):
    return build_facts(score_applicant(bundle, synthetic_applicant()))


def test_guardrails_reject_unsupported_claims(bundle):
    facts = _facts(bundle)
    assert any("number" in p for p in check_llm_output("Your rate will be 3.9%.", facts))
    assert any("protected" in p for p in check_llm_output(template_explanation(facts) + " Your age mattered.", facts))
    assert check_llm_output(template_explanation(facts), facts) == []


def test_llm_falls_back_to_template(bundle, monkeypatch):
    facts = _facts(bundle)
    monkeypatch.setenv("LLM_EXPLANATIONS", "1")
    bad = llm_explanation(facts, lambda prompt: "Great news, approved for 99999!")
    assert bad["source"] == "template" and bad["problems"]
    broken = llm_explanation(facts, lambda prompt: (_ for _ in ()).throw(RuntimeError("timeout")))
    assert broken["source"] == "template"


def test_llm_not_used_unless_enabled(bundle):
    os.environ.pop("LLM_EXPLANATIONS", None)
    out = llm_explanation(_facts(bundle), lambda prompt: "anything")
    assert out["source"] == "template"


def test_service_matches_pipeline_for_the_same_loans(bundle, population):
    """Training/serving check: a loan sent through the API path gets the same PD as in the pipeline."""
    import numpy as np
    import pandas as pd
    from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from credit_risk.service import applicant_frame, predict_pd

    pop, _ = population
    rows = pop.sample(40, random_state=3)
    derived = {"fico_mid", "credit_history_months", "loan_to_income"}
    served = []
    for _, r in rows.iterrows():
        a = {c: (None if pd.isna(r[c]) else r[c]) for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in derived}
        a.update(fico_range_low=r["fico_range_low"], fico_range_high=r["fico_range_high"],
                 earliest_cr_line=f"{r['earliest_cr_date']:%Y-%m}", application_date=f"{r['issue_date']:%Y-%m-%d}")
        served.append(predict_pd(bundle, applicant_frame(a))[0])
    np.testing.assert_allclose(served, predict_pd(bundle, rows), rtol=1e-9, atol=1e-12)
