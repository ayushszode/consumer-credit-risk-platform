"""Smoke tests for the Streamlit app: every page and every example applicant runs without an error.

Uses the committed model bundle and results, so it checks the app exactly as it is deployed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
PAGES = ["Score an applicant", "Compare the models", "Lending policy", "Monitoring", "Ask the data",
         "About the project"]

needs_results = pytest.mark.skipif(
    not (ROOT / "artifacts" / "model_bundle.joblib").exists() or not (ROOT / "reports" / "results" / "summary.json").exists(),
    reason="no committed model bundle and results")


def started_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def no_problems(at: AppTest, where: str) -> None:
    assert not at.exception, (where, [e.message for e in at.exception])
    assert not at.error, (where, [e.value for e in at.error])


@needs_results
def test_every_page_renders():
    at = started_app()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        no_problems(at, page)


@needs_results
def test_presets_fill_the_form_and_rank_sensibly():
    at = started_app()
    risk = {}
    for preset in ["strong", "typical", "stretched"]:
        at.button(key=f"preset_{preset}").click().run()
        no_problems(at, preset)
        risk[preset] = at.session_state["last_pd"]
        html = " ".join(m.value for m in at.markdown)
        assert 'class="verdict"' in html
    assert at.slider(key="f_fico").value == 675  # the stretched applicant was applied last
    assert risk["strong"] < risk["typical"] < risk["stretched"], risk


@needs_results
def test_editing_the_application_rescores():
    at = started_app()
    before = at.session_state["last_pd"]
    at.slider(key="f_fico").set_value(820).run()
    no_problems(at, "fico change")
    assert at.session_state["last_pd"] < before


@needs_results
def test_policy_cutoff_slider():
    at = started_app()
    at.sidebar.radio[0].set_value("Lending policy").run()
    for value in (5, 20, 35):
        at.slider(key="cutoff").set_value(value).run()
        no_problems(at, f"cut-off {value}")


@needs_results
def test_ask_page_without_a_key_shows_demo_or_explains():
    at = started_app()
    at.sidebar.radio[0].set_value("Ask the data").run()
    no_problems(at, "ask page")
    if (ROOT / "reports" / "results" / "agent_eval.csv").exists():
        pick = at.selectbox(key="ask_pick")
        pick.set_value(pick.options[-1]).run()
        no_problems(at, "demo mode")
        assert at.chat_message
    else:
        assert any("GEMINI_API_KEY" in i.value for i in at.info)
