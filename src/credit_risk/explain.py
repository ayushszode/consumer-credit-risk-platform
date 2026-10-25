"""Plain-English explanations of a decision.

Order of responsibility:
  1. The policy engine makes the decision (approve / refer / decline). Never the language model.
  2. Reason codes come from the scorecard points (deterministic, repeatable).
  3. `template_explanation` turns those facts into a short message. This is the default.
  4. Optionally, `llm_explanation` asks a language model to reword the SAME facts more naturally,
     then checks the answer before using it. If any check fails we fall back to the template.

The checks are guardrails, not a guarantee: they catch new numbers, unknown factors and banned
topics, but a reworded sentence can still be misleading, so the template remains the fallback.
Only the structured facts below are ever sent to a model - never raw borrower records.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

from .features import FEATURE_LABELS

DECISION_TEXT = {
    "approve": "approved",
    "refer": "passed to a member of our team for review",
    "decline": "declined",
}
BANNED_TOPICS = re.compile(r"\b(race|ethnic|religio|gender|sex|age|pregnan|disab|nationality|married|marital)\w*",
                           re.IGNORECASE)


def build_facts(decision: dict) -> dict:
    """The only information an LLM is allowed to see."""
    return {
        "policy_outcome": decision["policy_outcome"],
        "risk_band": decision["risk_band"],
        "probability_of_default_pct": round(100 * decision["probability_of_default"], 1),
        "model_version": decision["model_version"],
        "verified_factors": [
            {"feature": r["feature"], "factor": FEATURE_LABELS.get(r["feature"], r["feature"]),
             "applicant_value": _fmt(r["value"])}
            for r in decision["reason_codes"]
        ],
        "limitations": ["Educational model trained on historical US loans; not a real lending decision."],
    }


PHRASES = {
    "fico_mid": "a credit score of {v}", "dti": "a debt-to-income ratio of {v}%",
    "revol_util": "{v}% of available revolving credit in use", "bc_util": "{v}% of bank card limits in use",
    "inq_last_6mths": "{v} credit enquiries in the last six months", "loan_to_income": "a loan that is {v} times annual income",
    "annual_inc": "an annual income of {v}", "loan_amnt": "a loan amount of {v}",
    "acc_open_past_24mths": "{v} accounts opened in the last two years", "delinq_2yrs": "{v} late payments in the last two years",
    "credit_history_months": "a credit history of {v} months", "emp_length_years": "{v} years in the current job",
    "mths_since_recent_inq": "{v} months since the last credit enquiry", "percent_bc_gt_75": "{v}% of bank cards over 75% used",
    "home_ownership": "home ownership status ({v})", "purpose": "the loan purpose ({v})",
    "verification_status": "income verification status ({v})", "pub_rec_bankruptcies": "{v} recorded bankruptcies",
}


def template_explanation(facts: dict) -> str:
    text = (f"Your application has been {DECISION_TEXT[facts['policy_outcome']]}. "
            f"Our model puts it in risk band {facts['risk_band']}.")
    factors = facts["verified_factors"]
    if factors and facts["policy_outcome"] != "approve":
        parts = [PHRASES.get(f["feature"], f["factor"].lower() + " of {v}").format(v=f["applicant_value"]) for f in factors]
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        text += f" The main things that raised the estimated risk were {joined}."
    return text


def check_llm_output(text: str, facts: dict) -> list[str]:
    """Return a list of problems (empty list = passed)."""
    problems = []
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", json.dumps(facts)))
    for num in re.findall(r"\d+(?:\.\d+)?", text):
        if num not in allowed_numbers:
            problems.append(f"number not in the facts: {num}")
    if BANNED_TOPICS.search(text):
        problems.append("mentions a protected characteristic")
    outcome_word = DECISION_TEXT[facts["policy_outcome"]].split()[0]
    if outcome_word.lower() not in text.lower():
        problems.append("does not state the actual outcome")
    for other, word in DECISION_TEXT.items():
        if other != facts["policy_outcome"] and re.search(rf"\b{word.split()[0]}\b", text, re.IGNORECASE):
            problems.append(f"mentions a different outcome: {other}")
    if len(text) > 600:
        problems.append("too long")
    return problems


def llm_explanation(facts: dict, call_llm: Callable[[str], str] | None = None) -> dict:
    """Try the LLM if one is configured; otherwise (or on any failed check) use the template."""
    fallback = template_explanation(facts)
    if call_llm is None or os.getenv("LLM_EXPLANATIONS") != "1":
        return {"text": fallback, "source": "template", "problems": []}
    prompt = (
        "Rewrite the loan decision below for the applicant in at most three short, polite sentences.\n"
        "Rules: use ONLY these facts; state the outcome exactly; do not add reasons, numbers, advice or "
        "anything about personal characteristics.\n\nFACTS:\n" + json.dumps(facts, indent=2)
    )
    try:
        text = call_llm(prompt).strip()
    except Exception as exc:  # network issues etc. should never break a decision
        return {"text": fallback, "source": "template", "problems": [f"LLM call failed: {exc}"]}
    problems = check_llm_output(text, facts)
    if problems:
        return {"text": fallback, "source": "template", "problems": problems}
    return {"text": text, "source": "llm", "problems": []}


def _fmt(v) -> str:
    if v is None:
        return "not reported"
    if isinstance(v, float):
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:.2f}" if abs(v) < 10 else f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v).replace("_", " ").lower() if isinstance(v, str) else str(v)
