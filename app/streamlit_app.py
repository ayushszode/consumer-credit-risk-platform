"""Streamlit app: score a made-up applicant and explore the results of the full run.

    streamlit run app/streamlit_app.py

Reads the model bundle from artifacts/ and the tables from reports/results/, both written by
`python -m credit_risk.pipeline`. Applicants entered here are made up; nothing is stored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.features import FEATURE_LABELS  # noqa: E402
from credit_risk.service import applicant_frame, load_bundle, predict_pd, score_applicant  # noqa: E402
from credit_risk.synthetic import synthetic_applicant  # noqa: E402

RESULTS = ROOT / "reports" / "results"
GITHUB = "https://github.com/ayushszode/consumer-credit-risk-platform"
LINKEDIN = "https://www.linkedin.com/in/ayushsanjayzode"

st.set_page_config(page_title="Credit Desk · consumer credit risk", page_icon="◆", layout="wide",
                   initial_sidebar_state="auto")

# ---------------------------------------------------------------- look and feel
INK, SLATE, LINE, CREAM, PAPER = "#1C2A26", "#5E6E68", "#E8E2D6", "#FBF8F3", "#F3EEE4"
EMERALD, MINT, GOLD, AMBER, CORAL = "#127C5A", "#E3F4EC", "#C8902E", "#E3A23B", "#D2554D"
OUTCOME = {"approve": (EMERALD, "Approve"), "refer": (AMBER, "Refer to an underwriter"),
           "decline": (CORAL, "Decline")}
BAND_COLOUR = {"A": "#127C5A", "B": "#5AA67F", "C": "#C9A23F", "D": "#E0873A", "E": "#D2554D"}
MODEL_COLOUR = {"scorecard": "#4C7A93", "gbm": EMERALD, "benchmark": "#A69F92", "xgboost": GOLD, "torch_mlp": "#7D5A84"}
SHORT_NAME = {"scorecard": "Scorecard", "gbm": "Gradient boosting", "benchmark": "Benchmark"}
MODEL_NAME = {"scorecard": "Scorecard", "gbm": "Gradient boosting", "benchmark": "Benchmark (with LendingClub grade)",
              "xgboost": "XGBoost", "torch_mlp": "PyTorch network"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stMarkdown,
[data-testid="stWidgetLabel"] {font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif !important;}
header[data-testid="stHeader"] {background: transparent;}
.stAppDeployButton, [data-testid="stAppDeployButton"] {display: none;}
.stApp {
  background:
    radial-gradient(1100px 520px at 108% -12%, rgba(200,144,46,.13), transparent 60%),
    radial-gradient(900px 620px at -12% 112%, rgba(18,124,90,.10), transparent 60%),
    #FBF8F3;
  color: #1C2A26;
}
.block-container {max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}
h1, h2, h3 {font-family: 'Fraunces', Georgia, serif; color: #1C2A26; letter-spacing: -.01em;}

/* sidebar */
section[data-testid="stSidebar"] {background: #FFFDF9; border-right: 1px solid #E8E2D6;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: .52rem .75rem; border-radius: 12px; margin: .12rem 0; transition: background .15s ease;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: #F3EEE4;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background: #E3F4EC; box-shadow: inset 3px 0 0 #127C5A;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {color: #127C5A; font-weight: 700;}
.brand {display: flex; gap: .75rem; align-items: center; margin: .1rem 0 1.3rem;}
.brand-mark {width: 42px; height: 42px; border-radius: 13px; background: linear-gradient(135deg, #127C5A, #0A523C);
  display: grid; place-items: center; color: #fff; font-family: 'Fraunces', serif; font-size: 1.1rem; font-weight: 650;
  box-shadow: 0 6px 16px rgba(18,124,90,.25);}
.brand-name {font-family: 'Fraunces', serif; font-size: 1.22rem; font-weight: 650; color: #1C2A26; line-height: 1.1;}
.brand-sub {font-size: .78rem; color: #5E6E68;}
.side-card {background: #fff; border: 1px solid #E8E2D6; border-radius: 14px; padding: .85rem .95rem; margin-top: 1.1rem;
  font-size: .82rem; color: #5E6E68; line-height: 1.55;}
.side-card b {color: #1C2A26;}
.side-card a {color: #127C5A; font-weight: 600; text-decoration: none;}
.live-dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #127C5A; margin-right: .4rem;
  box-shadow: 0 0 0 4px rgba(18,124,90,.15);}

/* headings */
.eyebrow {font-size: .74rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: #C8902E; margin-bottom: .4rem;}
.page-title {font-family: 'Fraunces', Georgia, serif; font-size: 2.45rem; line-height: 1.08; font-weight: 650; color: #1C2A26; margin: 0;}
.page-title em {font-style: italic; color: #127C5A;}
.page-lead {font-size: 1.03rem; color: #5E6E68; max-width: 800px; margin: .65rem 0 1.4rem; line-height: 1.6;}
.section-title {font-family: 'Fraunces', Georgia, serif; font-size: 1.38rem; font-weight: 650; color: #1C2A26; margin: 1.7rem 0 .15rem;}
.section-sub {color: #5E6E68; font-size: .93rem; margin-bottom: .7rem; max-width: 820px; line-height: 1.5;}

/* cards */
.kpi {background: #fff; border: 1px solid #E8E2D6; border-radius: 16px; padding: .95rem 1.05rem 1rem;
  box-shadow: 0 1px 2px rgba(28,42,38,.04), 0 8px 22px rgba(28,42,38,.05); height: 100%;}
.kpi-label {font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #5E6E68;}
.kpi-value {font-family: 'Fraunces', Georgia, serif; font-size: 2.05rem; font-weight: 650; color: #1C2A26; line-height: 1.15; margin-top: .25rem;}
.kpi-note {font-size: .83rem; color: #5E6E68; margin-top: .2rem; line-height: 1.4;}
.kpi.green {border-top: 3px solid #127C5A;} .kpi.gold {border-top: 3px solid #C8902E;}
.kpi.coral {border-top: 3px solid #D2554D;} .kpi.slate {border-top: 3px solid #4C7A93;}
.decision {border-radius: 18px; padding: 1.15rem 1.3rem; color: #fff; display: flex; justify-content: space-between;
  align-items: center; gap: 1rem; box-shadow: 0 10px 26px rgba(28,42,38,.12);}
.decision .label {font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; opacity: .85;}
.decision .verdict {font-family: 'Fraunces', Georgia, serif; font-size: 1.95rem; font-weight: 650; line-height: 1.1;}
.decision .meta {text-align: right; font-size: .9rem; line-height: 1.5; opacity: .95;}
.quote {background: #E3F4EC; border-radius: 16px; padding: 1rem 1.15rem; color: #16483A; font-size: .98rem; line-height: 1.6;
  border-left: 4px solid #127C5A;}
.note {background: #FFF7E6; border: 1px solid #F1DFB8; border-radius: 14px; padding: .85rem 1rem; color: #6B4E16;
  font-size: .92rem; line-height: 1.55;}
.winner {background: linear-gradient(120deg, #fff 0%, #F2FAF6 100%); border: 1px solid #CDE8DA; border-radius: 18px;
  padding: 1.1rem 1.25rem; box-shadow: 0 8px 22px rgba(18,124,90,.08);}
.winner .tag {display: inline-block; background: #127C5A; color: #fff; border-radius: 999px; padding: .2rem .65rem;
  font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;}
.winner .name {font-family: 'Fraunces', Georgia, serif; font-size: 1.6rem; font-weight: 650; color: #1C2A26; margin: .45rem 0 .25rem;}
.winner p {color: #3E5049; margin: 0; line-height: 1.55;}
.chip {display: inline-block; padding: .25rem .65rem; border-radius: 999px; background: #F3EEE4; color: #1C2A26;
  font-size: .8rem; font-weight: 600; margin: 0 .35rem .4rem 0;}
.steps {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem;}
.step {background: #fff; border: 1px solid #E8E2D6; border-radius: 16px; padding: .95rem 1rem;}
.step .n {font-family: 'Fraunces', Georgia, serif; color: #C8902E; font-size: 1.35rem; font-weight: 650;}
.step .t {font-weight: 700; color: #1C2A26; margin: .15rem 0 .25rem;}
.step .d {font-size: .87rem; color: #5E6E68; line-height: 1.5;}
.limits {background: #fff; border: 1px solid #E8E2D6; border-radius: 16px; padding: 1rem 1.2rem; color: #3E5049; line-height: 1.6;}

/* streamlit widgets */
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
div[data-testid="stExpander"] details {border-radius: 14px; border: 1px solid #E8E2D6; background: #fff;}
.stButton button {border-radius: 999px; border: 1px solid #E8E2D6; background: #fff; color: #1C2A26; font-weight: 600; padding: .35rem 1rem;}
.stButton button:hover {border-color: #127C5A; color: #127C5A; background: #F7FCF9;}
div[data-testid="stDataFrame"] {border: 1px solid #E8E2D6; border-radius: 14px; overflow: hidden;}
@media (max-width: 900px) {.steps {grid-template-columns: 1fr 1fr;} .page-title {font-size: 1.85rem;}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
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


# ---------------------------------------------------------------- small helpers
def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%"


def _show(v) -> str:
    """Applicant values without trailing decimals (677.0 -> 677, 31.5 -> 31.5)."""
    if v is None or (isinstance(v, float) and v != v):
        return "none on file"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def _num(x: str) -> str:
    v = float(x)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:g}"


def pretty_bin(b: str) -> str:
    """'[-inf, 667)' -> 'under 667', '[667, 677)' -> '667 to 677', '[732, inf)' -> '732+'."""
    b = str(b)
    if not (b.startswith("[") and b.endswith(")") and "," in b):
        return b.replace("_", " ").capitalize()
    lo, hi = (t.strip() for t in b[1:-1].split(",", 1))
    if lo == "-inf":
        return f"under {_num(hi)}"
    if hi == "inf":
        return f"{_num(lo)}+"
    return f"{_num(lo)} to {_num(hi)}"


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def heading(eyebrow: str, title: str, lead: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
                f'<p class="page-lead">{lead}</p>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items: (label, value, note, tone) with tone in green / gold / coral / slate."""
    for col, (lab, value, note, tone) in zip(st.columns(len(items)), items):
        size = ' style="font-size:1.5rem"' if len(str(value)) > 12 else ""
        col.markdown(f'<div class="kpi {tone}"><div class="kpi-label">{lab}</div><div class="kpi-value"{size}>{value}</div>'
                     f'<div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


_charts = {"n": 0}


def show(fig: go.Figure, height: int = 360, margin: dict | None = None) -> None:
    _charts["n"] += 1
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans, sans-serif", color=INK, size=13), height=height,
                      margin=margin or dict(l=8, r=8, t=36, b=8), hoverlabel=dict(bgcolor="white", font_family="DM Sans"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#EFE9DD", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EFE9DD", zeroline=False, linecolor=LINE)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"chart_{_charts['n']}")


# ---------------------------------------------------------------- sidebar
PAGES = ["Score an applicant", "Compare the models", "Lending policy", "Monitoring", "About the project"]
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">CD</div><div><div class="brand-name">Credit Desk</div>'
                '<div class="brand-sub">Consumer credit risk platform</div></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown(
        f'<div class="side-card"><span class="live-dot"></span><b>Live model</b> · {MODEL_NAME[bundle["champion"]]}<br>'
        f'Version {bundle["model_version"]} · policy v{bundle["policy"]["version"]}<br>'
        f'Trained on {bundle.get("train_window", "2012-2013")} loans, tested on 2015.<br>'
        f'Data: {info["run_label"]}</div>'
        f'<div class="side-card">Built by <b>Ayush Sanjay Zode</b><br>'
        f'<a href="{GITHUB}" target="_blank">Code on GitHub</a> · <a href="{LINKEDIN}" target="_blank">LinkedIn</a></div>',
        unsafe_allow_html=True)
if info.get("synthetic"):
    st.warning("This demo is running on synthetic data. The numbers show how the system works, not real credit risk.")


# ================================================================ 1. score an applicant
PRESETS = {
    "strong": dict(loan_amnt=8000, annual_inc=95000, dti=9.0, fico_range_low=770, revol_util=18.0, bc_util=15.0,
                   inq_last_6mths=0, earliest_cr_line="2001-03", emp_length_years=10, home_ownership="MORTGAGE",
                   verification_status="Not Verified", purpose="credit_card", open_acc=10, total_acc=30,
                   acc_open_past_24mths=2, num_actv_rev_tl=3, mort_acc=2, delinq_2yrs=0, pub_rec=0,
                   pub_rec_bankruptcies=0, mo_sin_rcnt_tl=14, mths_since_recent_inq=18, revol_bal=6000,
                   total_rev_hi_lim=45000, tot_cur_bal=260000, percent_bc_gt_75=0.0),
    "typical": dict(loan_amnt=10000, annual_inc=60000, dti=18.0, fico_range_low=700, revol_util=50.0, bc_util=55.0,
                    inq_last_6mths=1, earliest_cr_line="2008-05", emp_length_years=5, home_ownership="MORTGAGE",
                    verification_status="Source Verified", purpose="debt_consolidation", open_acc=11, total_acc=24,
                    acc_open_past_24mths=4, num_actv_rev_tl=5, mort_acc=1, delinq_2yrs=0, pub_rec=0,
                    pub_rec_bankruptcies=0, mo_sin_rcnt_tl=6, mths_since_recent_inq=5, revol_bal=12000,
                    total_rev_hi_lim=25000, tot_cur_bal=120000, percent_bc_gt_75=33.0),
    "stretched": synthetic_applicant(),
}
INT_FIELDS = ["open_acc", "total_acc", "acc_open_past_24mths", "num_actv_rev_tl", "mort_acc", "delinq_2yrs", "pub_rec",
              "pub_rec_bankruptcies", "mo_sin_rcnt_tl", "mths_since_recent_inq"]
MONEY_FIELDS = ["revol_bal", "total_rev_hi_lim", "tot_cur_bal", "percent_bc_gt_75"]
HOMES = ["RENT", "MORTGAGE", "OWN"]
VERIFICATION = ["Not Verified", "Source Verified", "Verified"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car",
            "medical", "other"]


def apply_preset(name: str) -> None:
    p = PRESETS[name]
    ss = st.session_state
    ss["f_loan_amnt"], ss["f_annual_inc"] = int(p["loan_amnt"]), int(p["annual_inc"])
    ss["f_dti"], ss["f_fico"] = float(p["dti"]), int(p["fico_range_low"])
    ss["f_revol_util"], ss["f_bc_util"] = float(p["revol_util"]), float(p["bc_util"])
    ss["f_inq_last_6mths"], ss["f_earliest_cr_line"] = int(p["inq_last_6mths"]), str(p["earliest_cr_line"])
    ss["f_emp_length_years"] = int(p["emp_length_years"])
    ss["f_home_ownership"] = p["home_ownership"] if p["home_ownership"] in HOMES else HOMES[0]
    ss["f_verification_status"] = p["verification_status"] if p["verification_status"] in VERIFICATION else VERIFICATION[1]
    ss["f_purpose"] = p["purpose"] if p["purpose"] in PURPOSES else "other"
    for k in INT_FIELDS:
        ss[f"f_{k}"] = int(p.get(k) or 0)
    for k in MONEY_FIELDS:
        ss[f"f_{k}"] = float(p.get(k) or 0)
    ss["preset"] = name


if "f_loan_amnt" not in st.session_state:
    apply_preset("stretched")


def risk_gauge(pd_value: float, policy: dict) -> go.Figure:
    approve, decline = 100 * policy["approve_max_pd"], 100 * policy["decline_min_pd"]
    top = max(50.0, 100 * pd_value + 5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * pd_value,
        number=dict(suffix="%", valueformat=".1f", font=dict(family="Fraunces, Georgia, serif", size=46, color=INK)),
        title=dict(text="Chance this loan goes bad over 36 months", font=dict(size=14, color=SLATE)),
        gauge=dict(axis=dict(range=[0, top], ticksuffix="%", tickcolor=SLATE), bar=dict(color=INK, thickness=0.22),
                   bgcolor="white", borderwidth=0,
                   steps=[dict(range=[0, approve], color="#D7EFE3"), dict(range=[approve, decline], color="#F8E6C2"),
                          dict(range=[decline, top], color="#F5D2CF")],
                   threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=100 * pd_value))))
    return fig


def fico_curve(applicant: dict) -> pd.DataFrame:
    scores = list(range(600, 851, 10))
    frames = [applicant_frame({**applicant, "fico_range_low": f, "fico_range_high": f + 4}) for f in scores]
    return pd.DataFrame({"fico": scores, "pd": predict_pd(bundle, pd.concat(frames, ignore_index=True))})


if page == PAGES[0]:
    heading("Loan desk", "Would we lend to <em>this</em> applicant?",
            "Fill in a made-up application, or start from one of the three below. The model estimates the chance "
            "the loan goes bad over its 36 months, and the lending policy turns that into a decision. Nothing you "
            "type is stored.")
    b1, b2, b3, _ = st.columns([1.15, 1.15, 1.15, 2])
    b1.button("Strong applicant", key="preset_strong", on_click=apply_preset, args=("strong",), width="stretch")
    b2.button("Typical applicant", key="preset_typical", on_click=apply_preset, args=("typical",), width="stretch")
    b3.button("Stretched applicant", key="preset_stretched", on_click=apply_preset, args=("stretched",), width="stretch")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="margin-top:0">The application</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.number_input("Loan amount ($)", 500, 40000, key="f_loan_amnt", step=500)
            c2.number_input("Annual income ($)", 1000, 1_000_000, key="f_annual_inc", step=1000)
            st.slider("Credit score (FICO)", 600, 850, key="f_fico", step=5)
            st.slider("Debt-to-income ratio (%)", 0.0, 60.0, key="f_dti", step=0.5, format="%.1f")
            c1, c2 = st.columns(2)
            c1.slider("Revolving credit used (%)", 0.0, 120.0, key="f_revol_util", step=1.0, format="%.0f")
            c2.slider("Bank card credit used (%)", 0.0, 120.0, key="f_bc_util", step=1.0, format="%.0f")
            c1, c2 = st.columns(2)
            c1.number_input("Credit enquiries, last 6 months", 0, 20, key="f_inq_last_6mths")
            c2.text_input("First credit account (YYYY-MM)", key="f_earliest_cr_line")
            st.slider("Years in current job", 0, 10, key="f_emp_length_years")
            c1, c2 = st.columns(2)
            c1.selectbox("Home ownership", HOMES, key="f_home_ownership")
            c2.selectbox("Income verification", VERIFICATION, key="f_verification_status")
            st.selectbox("Purpose of the loan", PURPOSES, key="f_purpose", format_func=lambda s: s.replace("_", " ").capitalize())
            with st.expander("More of the credit file"):
                c1, c2 = st.columns(2)
                for i, k in enumerate(INT_FIELDS):
                    (c1 if i % 2 == 0 else c2).number_input(label(k), 0, 200, key=f"f_{k}")
                for i, k in enumerate(MONEY_FIELDS):
                    (c1 if i % 2 == 0 else c2).number_input(label(k), 0.0, 5_000_000.0, key=f"f_{k}")

    ss = st.session_state
    a = {"loan_amnt": ss["f_loan_amnt"], "annual_inc": ss["f_annual_inc"], "dti": ss["f_dti"],
         "fico_range_low": ss["f_fico"], "fico_range_high": ss["f_fico"] + 4, "revol_util": ss["f_revol_util"],
         "bc_util": ss["f_bc_util"], "inq_last_6mths": ss["f_inq_last_6mths"],
         "earliest_cr_line": ss["f_earliest_cr_line"], "emp_length_years": ss["f_emp_length_years"],
         "home_ownership": ss["f_home_ownership"], "verification_status": ss["f_verification_status"],
         "purpose": ss["f_purpose"], "mths_since_last_delinq": None,
         **{k: ss[f"f_{k}"] for k in INT_FIELDS + MONEY_FIELDS}}

    with right:
        try:
            r = score_applicant(bundle, a)
        except Exception as exc:  # show input problems instead of crashing the page
            st.error(f"Couldn't score this applicant: {exc}")
            st.stop()
        ss["last_pd"] = r["probability_of_default"]
        colour, verdict = OUTCOME[r["policy_outcome"]]
        st.markdown(
            f'<div class="decision" style="background:linear-gradient(120deg,{colour},{colour}DD)">'
            f'<div><div class="label">Decision</div><div class="verdict">{verdict}</div></div>'
            f'<div class="meta">Risk band <b>{r["risk_band"]}</b><br>Scorecard points <b>{r["scorecard_points"]:.0f}</b></div>'
            f'</div>', unsafe_allow_html=True)
        show(risk_gauge(r["probability_of_default"], bundle["policy"]), height=290, margin=dict(l=40, r=50, t=60, b=10))
        st.markdown(f'<div class="section-sub" style="text-align:center;margin-top:-.6rem">Green: approve below '
                    f'{pct(bundle["policy"]["approve_max_pd"], 0)} · amber: refer · red: decline from '
                    f'{pct(bundle["policy"]["decline_min_pd"], 0)}</div>', unsafe_allow_html=True)

        section("What pushed the risk up")
        reasons = r["reason_codes"]
        if reasons:
            gbm = "pd_reduction_if_typical" in reasons[0]
            rows = pd.DataFrame([{"factor": f'{label(x["feature"])}: {_show(x["value"])}',
                                  "size": 100 * x["pd_reduction_if_typical"] if gbm else x["points_below_best"]}
                                 for x in reasons])[::-1]
            fig = go.Figure(go.Bar(x=rows["size"], y=rows["factor"], orientation="h", marker_color=GOLD,
                                   text=[f"{v:.1f} pts" if gbm else f"{v:.0f} pts" for v in rows["size"]],
                                   textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>%{x:.2f}<extra></extra>"))
            fig.update_xaxes(title="Risk would fall by (percentage points) if this were typical" if gbm
                             else "Points below the best possible", showgrid=False)
            show(fig, height=210, margin=dict(l=8, r=70, t=20, b=8))
        else:
            st.markdown('<div class="quote">Nothing in this application stands out as raising the risk.</div>',
                        unsafe_allow_html=True)
        section("In plain English")
        st.markdown(f'<div class="quote">{r["explanation"]["text"]}</div>', unsafe_allow_html=True)
        st.caption(f"Explanation written from fixed templates; the decision comes from the policy. {r['target_definition']}")

    section("What if the credit score were different?",
            "Everything else in the application stays the same. The line shows how the estimated risk changes as "
            "the credit score moves; the dot is the score entered above.")
    curve = fico_curve(a)
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=100 * bundle["policy"]["approve_max_pd"], fillcolor="#D7EFE3", opacity=.55, line_width=0, layer="below")
    fig.add_hrect(y0=100 * bundle["policy"]["approve_max_pd"], y1=100 * bundle["policy"]["decline_min_pd"],
                  fillcolor="#F8E6C2", opacity=.55, line_width=0, layer="below")
    fig.add_hrect(y0=100 * bundle["policy"]["decline_min_pd"], y1=max(60, 100 * curve["pd"].max() + 5),
                  fillcolor="#F5D2CF", opacity=.55, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=curve["fico"], y=100 * curve["pd"], mode="lines", line=dict(color=INK, width=3),
                             name="Estimated risk", hovertemplate="FICO %{x}<br>Risk %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[ss["f_fico"]], y=[100 * r["probability_of_default"]], mode="markers",
                             marker=dict(size=14, color=colour, line=dict(color="white", width=3)), name="This applicant",
                             hovertemplate="This applicant<br>Risk %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="Chance of going bad (%)", range=[0, max(40, 100 * curve["pd"].max() + 5)])
    fig.update_xaxes(title="Credit score (FICO)")
    show(fig, height=330)


# ================================================================ 2. compare the models
elif page == PAGES[1]:
    m = table("metrics")
    test = m[m["split"] == "test"].set_index("model")
    heading("Model room", "Two models, one rule, <em>one winner</em>",
            "An explainable scorecard against gradient boosting, plus a benchmark that is allowed to see "
            "LendingClub's own grade. The rule for choosing was written down before the comparison and judged on "
            "2014; every number here is from 2015, a year no model learned from.")
    st.markdown(f'<div class="winner"><span class="tag">Champion</span><div class="name">{MODEL_NAME[info["champion"]]}</div>'
                f'<p>{info["champion_reason"]}</p></div>', unsafe_allow_html=True)
    st.write("")
    kpis([(SHORT_NAME[k], f"{test.loc[k, 'auc']:.3f}", f"AUC on 2015 · 95% CI {test.loc[k, 'auc_ci_low']:.3f}–"
           f"{test.loc[k, 'auc_ci_high']:.3f}", tone) for k, tone in [("scorecard", "slate"), ("gbm", "green"),
                                                                     ("benchmark", "gold")] if k in test.index]
         + [("Loans tested", f"{int(test['loans'].iloc[0]):,}", "the same loans for every model", "slate")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("How well each model ranks risk", "AUC on 2014 (used to choose) and 2015 (used to report), with "
                "95% bootstrap intervals.")
        fig = go.Figure()
        for split, name, colour in [("validation", "2014 (validation)", "#CFC6B6"), ("test", "2015 (test)", EMERALD)]:
            d = m[(m["split"] == split) & m["model"].isin(SHORT_NAME)]
            fig.add_trace(go.Bar(x=[SHORT_NAME[x] for x in d["model"]], y=d["auc"], name=name, marker_color=colour,
                                 error_y=dict(type="data", array=d["auc_ci_high"] - d["auc"],
                                              arrayminus=d["auc"] - d["auc_ci_low"], color=INK, thickness=1.2),
                                 hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>"))
        fig.update_yaxes(range=[0.6, 0.71], title="AUC")
        fig.update_layout(barmode="group", bargap=.35)
        show(fig, height=360)
    with c2:
        section("Do the probabilities match reality?", "2015 loans split into ten groups by predicted risk. Points "
                "above the dashed line mean more loans went bad than the model expected.")
        fig = go.Figure()
        top = 0.0
        for k in ["scorecard", "gbm"]:
            c = table(f"calibration_test_{k}")
            top = max(top, c["predicted_pd"].max(), c["actual_bad_rate"].max())
            fig.add_trace(go.Scatter(x=100 * c["predicted_pd"], y=100 * c["actual_bad_rate"], mode="lines+markers",
                                     name=MODEL_NAME[k], line=dict(color=MODEL_COLOUR[k], width=2.5),
                                     marker=dict(size=8), customdata=c["loans"],
                                     hovertemplate="Predicted %{x:.1f}%<br>Actual %{y:.1f}%<br>%{customdata:,} loans<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 100 * top], y=[0, 100 * top], mode="lines", name="Perfect match",
                                 line=dict(color="#A69F92", dash="dash")))
        fig.update_xaxes(title="Predicted chance of going bad (%)")
        fig.update_yaxes(title="Share that actually went bad (%)")
        show(fig, height=360)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("What the winning model leans on", "How much 2015 AUC drops when each input is shuffled.")
        imp = table("permutation_importance_gbm").head(10)[::-1]
        fig = go.Figure(go.Bar(x=imp["auc_drop"], y=[label(f) for f in imp["feature"]], orientation="h",
                               marker=dict(color=imp["auc_drop"], colorscale=[[0, "#CDE8DA"], [1, EMERALD]]),
                               hovertemplate="%{y}<br>AUC drop %{x:.4f}<extra></extra>"))
        fig.update_xaxes(title="Drop in AUC")
        show(fig, height=380)
    with c2:
        section("Inside the scorecard", "Pick a characteristic to see how the points change across its ranges. "
                "More points means lower risk.")
        pts = table("points_table")
        feats = list(dict.fromkeys(pts["feature"]))
        default = feats.index("fico_mid") if "fico_mid" in feats else 0
        f = st.selectbox("Characteristic", feats, index=default, format_func=label, label_visibility="collapsed")
        d = pts[pts["feature"] == f]
        fig = go.Figure(go.Bar(x=[pretty_bin(b) for b in d["bin"]], y=d["points"], customdata=d[["bad_rate", "n"]],
                               marker=dict(color=d["bad_rate"], colorscale=[[0, EMERALD], [0.5, GOLD], [1, CORAL]],
                                           colorbar=dict(title="Bad rate", tickformat=".0%", thickness=10)),
                               hovertemplate="%{x}<br>%{y} points<br>Bad rate %{customdata[0]:.1%}"
                                             "<br>%{customdata[1]:,} loans<extra></extra>"))
        fig.update_yaxes(title="Points")
        fig.update_xaxes(title=label(f), type="category")
        show(fig, height=330)

    extra = table("extra_challengers")
    trained = extra[extra["status"] == "trained"]
    if len(trained):
        vs = table("extra_challengers_vs_scorecard_test")
        section("Extra challengers", "Reported for comparison, never deployed.")
        st.dataframe(vs.round(4), hide_index=True, width="stretch")
    else:
        st.markdown('<div class="note"><b>XGBoost and a PyTorch network</b> are set up as extra challengers and '
                    'tested automatically, but were not part of this run. Their comparison will appear here after '
                    'the next full run.</div>', unsafe_allow_html=True)


# ================================================================ 3. lending policy
elif page == PAGES[2]:
    curve = table("threshold_curve_test")
    policy = bundle["policy"]
    heading("Credit policy", "Where should we <em>draw the line?</em>",
            "Every lender trades approvals against losses. Move the cut-off and see what it would have meant for "
            "the loans LendingClub actually funded in 2015. This is a what-if on past loans, not a forecast for "
            "every applicant.")
    cut = st.slider("Approve automatically when the predicted chance of going bad is at most (%)",
                    int(round(100 * curve["cutoff_pd"].min())), int(round(100 * curve["cutoff_pd"].max())),
                    int(round(100 * policy["approve_max_pd"])), 1, key="cutoff") / 100
    row = curve.iloc[(curve["cutoff_pd"] - cut).abs().argmin()]
    overall = table("splits").set_index("split").loc["test", "bad_rate"]
    kpis([("Loans approved", pct(row["selected_share"]), f"of 2015 loans at a {pct(cut, 0)} cut-off", "green"),
          ("Bad rate of approved", pct(row["observed_bad_rate"]), f"against {pct(overall)} for all 2015 loans", "gold"),
          ("Bad loans avoided", pct(row["bad_loans_avoided_share"]), "share of bad loans not approved", "coral")])

    section("The trade-off", f"Each point on the line is a cut-off. Moving right approves more loans, and the bad "
            f"rate among them climbs towards the {pct(overall)} of all 2015 loans.")
    fig = go.Figure()
    fig.add_hline(y=100 * overall, line=dict(color=SLATE, dash="dot"), annotation_text=f"All 2015 loans: {pct(overall)}",
                  annotation_position="bottom right", annotation_font_color=SLATE)
    fig.add_trace(go.Scatter(x=100 * curve["selected_share"], y=100 * curve["observed_bad_rate"], mode="lines",
                             line=dict(color=EMERALD, width=3.5), customdata=100 * curve["cutoff_pd"], name="Every cut-off",
                             fill="tozeroy", fillcolor="rgba(18,124,90,0.07)",
                             hovertemplate="Cut-off %{customdata:.0f}%<br>Approve %{x:.1f}% of loans"
                                           "<br>Bad rate %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[100 * row["selected_share"]], y=[100 * row["observed_bad_rate"]], mode="markers+text",
                             marker=dict(size=18, color=GOLD, line=dict(color="white", width=3)),
                             text=[f"<b>{pct(cut, 0)} cut-off</b>"], textposition="top left", name="Your cut-off",
                             hovertemplate="Your cut-off<br>Approve %{x:.1f}%<br>Bad rate %{y:.1f}%<extra></extra>"))
    fig.update_xaxes(title="Share of 2015 loans approved (%)", range=[0, 101])
    fig.update_yaxes(title="Bad rate among approved loans (%)", range=[0, 100 * curve["observed_bad_rate"].max() + 2])
    show(fig, height=380)

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        section("Risk bands", "Each band holds loans with similar predicted risk. Bars show what actually happened; "
                "diamonds show what the model expected.")
        bands = table("bands_test")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bands["band"], y=100 * bands["bad_rate"], name="Actual bad rate",
                             marker_color=[BAND_COLOUR.get(b, SLATE) for b in bands["band"]],
                             text=[pct(v) for v in bands["bad_rate"]], textposition="outside",
                             customdata=bands["loans"], hovertemplate="Band %{x}<br>Actual %{y:.1f}%<br>%{customdata:,} loans<extra></extra>"))
        fig.add_trace(go.Scatter(x=bands["band"], y=100 * bands["mean_predicted_pd"], name="Predicted",
                                 mode="markers", marker=dict(symbol="diamond", size=13, color=INK),
                                 hovertemplate="Band %{x}<br>Predicted %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(title="Bad rate (%)")
        show(fig, height=340)
    with c2:
        section("Decisions at the current policy", f"Approve below {pct(policy['approve_max_pd'], 0)}, decline at "
                f"{pct(policy['decline_min_pd'], 0)} or above, refer the rest.")
        out = table("policy_outcomes_test")
        fig = go.Figure(go.Pie(labels=[OUTCOME[o][1] for o in out["outcome"]], values=out["loans"], hole=.62,
                               marker=dict(colors=[OUTCOME[o][0] for o in out["outcome"]], line=dict(color="white", width=3)),
                               text=[pct(v) for v in out["observed_bad_rate"]], textinfo="percent", sort=False,
                               hovertemplate="%{label}<br>%{percent} of loans<br>Bad rate %{text}<extra></extra>"))
        fig.add_annotation(text=f"<b>{pct(info['approve_share_test'], 0)}</b><br>approved", showarrow=False,
                           font=dict(family="Fraunces, Georgia, serif", size=20, color=INK))
        show(fig, height=340)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        le = info["lgd_ead"]
        section("Expected loss", f"PD × LGD × EAD on approved loans. Estimated LGD {le['lgd']:.2f} and EAD "
                f"{le['ead_share']:.2f}, from {le['charged_off_loans_used']:,} charged-off training loans, give "
                f"<b>{pct(info['expected_loss_pct_approved_test'], 2)}</b> of the amount lent. The grid shows how "
                "that moves if the assumptions are off.")
        el = table("el_sensitivity").pivot(index="lgd", columns="ead_share", values="expected_loss_pct_of_lending")
        fig = go.Figure(go.Heatmap(z=100 * el.values, x=[f"EAD {v:.1f}" for v in el.columns],
                                   y=[f"LGD {v:.1f}" for v in el.index], text=[[f"{100 * v:.1f}%" for v in rowv] for rowv in el.values],
                                   texttemplate="%{text}", colorscale=[[0, "#EAF5EF"], [.5, "#E9C77D"], [1, CORAL]],
                                   showscale=False, hovertemplate="%{y}, %{x}<br>Expected loss %{text}<extra></extra>"))
        show(fig, height=300)
    with c2:
        section("Pricing by band", "An illustrative rate that just covers expected loss and funding, next to the "
                "rate LendingClub actually charged.")
        rate = table("rate_comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rate["band"], y=100 * rate["illustrative_rate"], name="Illustrative rate", marker_color=EMERALD))
        fig.add_trace(go.Bar(x=rate["band"], y=100 * rate["lendingclub_rate"], name="LendingClub's rate", marker_color="#CFC6B6"))
        fig.update_layout(barmode="group", bargap=.3)
        fig.update_yaxes(title="Interest rate (%)")
        show(fig, height=300)


# ================================================================ 4. monitoring
elif page == PAGES[3]:
    rep = table("monitoring_replay")
    drift = table("drift_test_vs_train").sort_values("psi", ascending=False)
    heading("Health check", "Is the model <em>still working?</em>",
            "A replay of what a monitoring team would have seen each quarter after the model was built. Shifts in "
            "the inputs show up straight away; whether loans really went bad takes up to 36 months to know.")
    top = drift.iloc[0]
    kpis([("Score stability", f"{rep['score_psi'].max():.3f}", "highest quarterly PSI (below 0.10 is stable)", "green"),
          ("Ranking power", f"{rep['auc'].min():.3f}–{rep['auc'].max():.3f}", "AUC range across the 8 quarters", "slate"),
          ("Input to watch", label(top["feature"]), f"PSI {top['psi']:.2f} between training years and 2015", "gold")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Has the score shifted?", "Population Stability Index of the score each quarter.")
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=0.10, fillcolor="#D7EFE3", opacity=.6, line_width=0, layer="below")
        fig.add_hrect(y0=0.10, y1=0.25, fillcolor="#F8E6C2", opacity=.6, line_width=0, layer="below")
        fig.add_hrect(y0=0.25, y1=0.30, fillcolor="#F5D2CF", opacity=.6, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=rep["quarter"], y=rep["score_psi"], mode="lines+markers", line=dict(color=INK, width=3),
                                 marker=dict(size=9), name="Score PSI", hovertemplate="%{x}<br>PSI %{y:.3f}<extra></extra>"))
        fig.update_yaxes(range=[0, 0.3], title="PSI")
        show(fig, height=330)
    with c2:
        section("Predicted against actual", "The gap that opens in 2015 is the model under-estimating risk as "
                "defaults rose: the ranking held, the level drifted.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rep["quarter"], y=100 * rep["observed_bad_rate"], name="Actual bad rate",
                                 mode="lines+markers", line=dict(color=CORAL, width=3)))
        fig.add_trace(go.Scatter(x=rep["quarter"], y=100 * rep["mean_predicted_pd"], name="Predicted",
                                 mode="lines+markers", line=dict(color=EMERALD, width=3, dash="dash")))
        fig.update_yaxes(title="Bad rate (%)")
        show(fig, height=330)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Bad rate by issue quarter", "Every quarter of loans in the project, coloured by how it was used.")
        v = table("vintages")
        year = v["issue_quarter"].str[:4]
        use = year.map(lambda y: "Training" if y in ("2012", "2013") else ("Validation" if y == "2014" else "Test"))
        colours = {"Training": "#CFC6B6", "Validation": GOLD, "Test": EMERALD}
        fig = go.Figure()
        for u in ["Training", "Validation", "Test"]:
            d = v[use == u]
            fig.add_trace(go.Bar(x=d["issue_quarter"], y=100 * d["bad_rate"], name=u, marker_color=colours[u],
                                 customdata=d["loans"], hovertemplate="%{x}<br>Bad rate %{y:.1f}%<br>%{customdata:,} loans<extra></extra>"))
        fig.update_yaxes(title="Bad rate (%)")
        show(fig, height=330)
    with c2:
        section("Which inputs moved most", "PSI of each input, 2015 against the training years.")
        d = drift.head(10)[::-1]
        status_colour = d["status"].map({"stable": "#9CCFB4", "investigate": GOLD}).fillna(CORAL)
        fig = go.Figure(go.Bar(x=d["psi"], y=[label(f) for f in d["feature"]], orientation="h", marker_color=status_colour,
                               customdata=d["status"], hovertemplate="%{y}<br>PSI %{x:.3f} (%{customdata})<extra></extra>"))
        fig.add_vline(x=0.10, line=dict(color=SLATE, dash="dot"))
        fig.update_xaxes(title="PSI")
        show(fig, height=330)


# ================================================================ 5. about
else:
    audit = table("population_audit").set_index("step")["loans"]
    heading("Behind the scenes", "From a raw loan file to <em>a lending decision</em>",
            "An independent project on public US loan data from LendingClub. It follows the steps a lender's risk "
            "team would take before trusting a model, and says plainly where it falls short.")
    steps = [("Population", "Clean the raw file with SQL and keep 36-month loans issued 2012–2015 with a known outcome."),
             ("Leakage checks", "A register of every column blocks anything only known after a loan was funded."),
             ("Two models", "A points-based scorecard and gradient boosting, plus a benchmark with the lender's own grade."),
             ("Champion rule", "Written in advance, judged on 2014, so 2015 stays a clean test."),
             ("Policy and loss", "Risk bands, approve / refer / decline cut-offs and expected loss (PD × LGD × EAD)."),
             ("Monitoring", "Quarterly stability (PSI), drift by input and a replay of performance over time."),
             ("Explanations", "Reason codes and a plain-English summary for every decision; the policy decides, not the text."),
             ("Delivery", "A FastAPI service, this app and automated tests that run on every change.")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{i:02d}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (t, d) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        section(f"From {audit['total_rows'] / 1e6:.2f} million rows to {int(audit['eligible']):,} loans",
                "Each rule, in order, and how many loans remained.")
        remaining, stages, values = int(audit["total_rows"]), ["All loans in the file"], [int(audit["total_rows"])]
        names = {"not_36_month_term": "36-month term only", "issued_outside_window": "Issued 2012–2015",
                 "not_individual_application": "Individual applications", "unresolved_outcome": "Known final outcome"}
        for step, name in names.items():
            remaining -= int(audit.get(step, 0))
            stages.append(name)
            values.append(remaining)
        fig = go.Figure(go.Funnel(y=stages, x=values, textinfo="value", texttemplate="%{value:,}",
                                  marker=dict(color=["#CFC6B6", "#B9C9B9", "#8DBFA3", "#4FA37A", EMERALD]),
                                  connector=dict(fillcolor="#F1ECE2", line=dict(color=LINE, width=1))))
        show(fig, height=340)
    with c2:
        section("What this is not")
        st.markdown(
            f'<div class="limits">• Not a model of UK borrowers: the data is from a US lender, 2012–2015.<br>'
            f'• Not a real lender\'s system, and not built to regulatory standards.<br>'
            f'• Not an IFRS 9 expected credit loss calculation.<br>'
            f'• Only loans that were approved are in the data, so nobody knows how declined applicants would have done.<br>'
            f'• Protected characteristics are not in the data, so fairness can\'t be fully checked.<br><br>'
            f'<b>Target:</b> {info["target_definition"]}</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("".join(f'<span class="chip">{t}</span>' for t in
                            ["Python", "SQL", "DuckDB", "PySpark", "scikit-learn", "Scorecards", "Gradient boosting",
                             "Model validation", "PSI monitoring", "FastAPI", "Streamlit", "GitHub Actions"]),
                    unsafe_allow_html=True)
        st.markdown(f'<div class="side-card">The full write-ups (model development, validation review, policy '
                    f'analysis, limitations) are in the <a href="{GITHUB}" target="_blank">GitHub repository</a>.</div>',
                    unsafe_allow_html=True)
