"""Streamlit demo.

    streamlit run app/streamlit_app.py

Reads the model bundle from artifacts/ and the tables from reports/results/, both written by
`python -m credit_risk.pipeline`. Applicants entered here are made up; nothing is stored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.features import FEATURE_LABELS  # noqa: E402
from credit_risk.service import load_bundle, score_applicant  # noqa: E402
from credit_risk.synthetic import synthetic_applicant  # noqa: E402

RESULTS, FIGS = ROOT / "reports" / "results", ROOT / "reports" / "figures"

st.set_page_config(page_title="Consumer credit risk platform", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


@st.cache_data
def summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


@st.cache_resource(show_spinner=False)
def train_demo_model() -> None:
    """No trained model yet (e.g. a fresh online deployment): build one on synthetic data."""
    from credit_risk.pipeline import run
    run(synthetic=True, engine="pandas")


if not (ROOT / "artifacts" / "model_bundle.joblib").exists() or not (RESULTS / "summary.json").exists():
    with st.spinner("First start: training the demo model on synthetic data. This takes about a minute..."):
        train_demo_model()

try:
    bundle, info = get_bundle(), summary()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m credit_risk.pipeline --synthetic` (or with the real data) first.")
    st.stop()

st.sidebar.title("Credit risk platform")
page = st.sidebar.radio("Page", ["Score an applicant", "Model comparison", "Lending policy", "Monitoring",
                                 "About and limitations"])
st.sidebar.caption(f"Model {bundle['model_version']} · policy v{bundle['policy']['version']}")
st.sidebar.caption(f"Data: {info['run_label']}")
if info.get("synthetic"):
    st.warning("This demo is running on synthetic data. The numbers show how the system works, "
               "not real credit risk.")

pct = lambda v: f"{v:.1%}"  # noqa: E731

# ---------------------------------------------------------------- score
if page == "Score an applicant":
    st.header("Score an applicant")
    st.caption("Enter a made-up applicant. The policy engine makes the decision; the text below only explains it.")
    d = synthetic_applicant()
    left, right = st.columns([1, 1.2])
    with left:
        a = {}
        a["loan_amnt"] = st.number_input("Loan amount", 500, 40000, d["loan_amnt"], step=500)
        a["annual_inc"] = st.number_input("Annual income", 1000, 1_000_000, d["annual_inc"], step=1000)
        a["dti"] = st.slider("Debt-to-income ratio (%)", 0.0, 60.0, d["dti"], 0.5)
        fico = st.slider("Credit score (FICO)", 600, 850, d["fico_range_low"], 5)
        a["fico_range_low"], a["fico_range_high"] = fico, fico + 4
        a["revol_util"] = st.slider("Revolving credit used (%)", 0.0, 120.0, d["revol_util"], 1.0)
        a["bc_util"] = st.slider("Bank card credit used (%)", 0.0, 120.0, d["bc_util"], 1.0)
        a["inq_last_6mths"] = st.number_input("Credit enquiries in last 6 months", 0, 20, d["inq_last_6mths"])
        a["earliest_cr_line"] = st.text_input("First credit account (YYYY-MM)", d["earliest_cr_line"])
        a["emp_length_years"] = st.slider("Years in current job", 0, 10, d["emp_length_years"])
        a["home_ownership"] = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
        a["verification_status"] = st.selectbox("Income verification", ["Not Verified", "Source Verified", "Verified"], 1)
        a["purpose"] = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
                                                "small_business", "car", "medical", "other"])
        with st.expander("More credit file details"):
            for k in ["open_acc", "total_acc", "acc_open_past_24mths", "num_actv_rev_tl", "mort_acc", "delinq_2yrs",
                      "pub_rec", "pub_rec_bankruptcies", "mo_sin_rcnt_tl", "mths_since_recent_inq"]:
                a[k] = st.number_input(FEATURE_LABELS.get(k, k), 0, 200, int(d[k] or 0))
            for k in ["revol_bal", "total_rev_hi_lim", "tot_cur_bal", "percent_bc_gt_75"]:
                a[k] = st.number_input(FEATURE_LABELS.get(k, k), 0.0, 5_000_000.0, float(d[k]))
        a["mths_since_last_delinq"] = None
    with right:
        try:
            r = score_applicant(bundle, a)
        except Exception as exc:  # show input problems instead of crashing the page
            st.error(f"Couldn't score this applicant: {exc}")
            st.stop()
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability of default", pct(r["probability_of_default"]))
        c2.metric("Risk band", r["risk_band"])
        c3.metric("Scorecard points", f"{r['scorecard_points']:.0f}")
        outcome = {"approve": st.success, "refer": st.warning, "decline": st.error}[r["policy_outcome"]]
        outcome(f"Policy outcome: **{r['policy_outcome']}**")
        st.subheader("Main reasons risk is higher")
        if r["reason_codes"]:
            st.table(pd.DataFrame([{"Factor": FEATURE_LABELS.get(x["feature"], x["feature"]), "Applicant": x["value"],
                                    **({"Points below best": x["points_below_best"]} if "points_below_best" in x else
                                       {"PD reduction if typical": pct(x["pd_reduction_if_typical"])})}
                                   for x in r["reason_codes"]]))
        else:
            st.write("No factor scored below the best possible points.")
        st.subheader("Plain-English summary")
        st.info(r["explanation"]["text"])
        st.caption(f"Explanation source: {r['explanation']['source']}. Target: {r['target_definition']}")

# ---------------------------------------------------------------- models
elif page == "Model comparison":
    st.header("Scorecard vs gradient boosting")
    st.write(f"**Champion: {info['champion']}.** {info['champion_reason']}")
    m = table("metrics")
    st.dataframe(m[m["split"] == "test"][["model", "loans", "bad_rate", "auc", "auc_ci_low", "auc_ci_high", "gini", "ks",
                                         "brier", "log_loss", "calibration_slope"]].round(3), hide_index=True)
    st.caption("All models are scored on the same 2015 loans. 'benchmark' also sees LendingClub's own grade and rate, "
               "so it is a reference point, not a candidate.")
    c1, c2 = st.columns(2)
    c1.image(str(FIGS / "roc_test.png"))
    c2.image(str(FIGS / "calibration_test.png"))
    st.subheader("What drives the scorecard")
    st.image(str(FIGS / "information_value.png"), width=520)
    with st.expander("Full points table"):
        st.dataframe(table("points_table")[["feature", "bin", "n", "bad_rate", "woe", "points"]].round(3), hide_index=True)

# ---------------------------------------------------------------- policy
elif page == "Lending policy":
    st.header("Approval rate vs credit risk")
    st.caption("Offline analysis on loans LendingClub actually funded in 2015. It shows a trade-off, not what would "
               "happen to all applicants under a new policy.")
    curve = table("threshold_curve_test")
    cut = st.slider("Approve if predicted PD is at most", float(curve["cutoff_pd"].min()), float(curve["cutoff_pd"].max()),
                    float(bundle["policy"]["approve_max_pd"]), 0.01)
    row = curve.iloc[(curve["cutoff_pd"] - cut).abs().argmin()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Loans selected", pct(row["selected_share"]))
    c2.metric("Bad rate of selected", pct(row["observed_bad_rate"]))
    c3.metric("Bad loans avoided", pct(row["bad_loans_avoided_share"]))
    st.image(str(FIGS / "policy_tradeoff.png"), width=560)
    st.subheader("Risk bands")
    st.dataframe(table("bands_test").round(3), hide_index=True)
    st.subheader("Expected loss sensitivity")
    st.write(f"LGD estimated at {info['lgd_ead']['lgd']:.2f} and exposure at default at {info['lgd_ead']['ead_share']:.2f} "
             f"of the loan amount, from charged-off training loans. Expected lifetime loss on auto-approved 2015 loans: "
             f"{pct(info['expected_loss_pct_approved_test'])} of the amount lent.")
    sens = table("el_sensitivity").pivot(index="lgd", columns="ead_share", values="expected_loss_pct_of_lending")
    st.dataframe((sens * 100).round(2).add_suffix(" EAD"), use_container_width=False)

# ---------------------------------------------------------------- monitoring
elif page == "Monitoring":
    st.header("Monitoring replay")
    st.caption("Each later quarter is compared with the training years. Distribution checks work from day one; "
               "AUC and bad rates need loans to finish first (up to 36 months).")
    rep = table("monitoring_replay")
    st.dataframe(rep.round(3), hide_index=True)
    st.image(str(FIGS / "monitoring_psi.png"))
    st.subheader("Bad rate by issue quarter")
    st.image(str(FIGS / "vintages.png"))
    with st.expander("Feature drift: 2015 vs training years"):
        st.dataframe(table("drift_test_vs_train").round(3), hide_index=True)

# ---------------------------------------------------------------- about
else:
    st.header("About this project")
    st.markdown(f"""
An independent, educational project using historical **US** consumer-loan data (LendingClub, 2012-2015,
36-month loans) to show how a lender could estimate repayment risk at application time, compare an
interpretable scorecard with a machine-learning model, and judge the trade-off between approvals and risk.

**Target:** {info['target_definition']}

**Please don't read this as:**
- a model of UK borrowers (the data is from the US, 2012-2015)
- a real lender's system or a regulatory-compliant model
- an IFRS 9 expected credit loss calculation
- evidence of how rejected applicants would have behaved (only funded loans are in the data)

**Known limitations:** the economy, products and borrowers have changed since 2015; the data only contains
loans that were approved; protected characteristics are not in the data, so fairness can't be fully assessed;
loss assumptions (LGD/EAD) are simple averages.

Full write-ups are in the `reports/` folder of the GitHub repository.
""")
