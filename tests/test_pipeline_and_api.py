import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.synthetic import synthetic_applicant


def test_pipeline_writes_everything(pipeline_run, cfg):
    reports = Path(cfg["data"]["reports_dir"])
    for name in ["metrics", "bands_test", "threshold_curve_test", "monitoring_replay", "population_audit",
                 "information_value", "feature_register"]:
        assert (reports / "results" / f"{name}.csv").exists(), name
    summary = json.loads((reports / "results" / "summary.json").read_text())
    assert summary["synthetic"] is True
    assert summary["champion"] in {"scorecard", "gbm"}
    assert "SYNTHETIC" in (reports / "results.md").read_text().upper()
    assert (Path(cfg["data"]["artifacts_dir"]) / "model_bundle.joblib").exists()


def test_models_compared_on_identical_loans(pipeline_run, cfg):
    m = pd.read_csv(Path(cfg["data"]["reports_dir"]) / "results" / "metrics.csv")
    for split, g in m.groupby("split"):
        assert g["loans"].nunique() == 1, split


def test_api_health_and_score(bundle, cfg, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MODEL_BUNDLE", str(Path(cfg["data"]["artifacts_dir"]) / "model_bundle.joblib"))
    import api.main as api_main
    api_main._bundle = None
    client = TestClient(api_main.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/score", json=synthetic_applicant())
    assert r.status_code == 200
    body = r.json()
    assert body["policy_outcome"] in {"approve", "refer", "decline"} and body["api_version"]
    bad = client.post("/score", json={**synthetic_applicant(), "dti": 500})
    assert bad.status_code == 422
    future_credit = client.post("/score", json={**synthetic_applicant(), "application_date": "2015-06-01",
                                                "earliest_cr_line": "2016-04"})
    assert future_credit.status_code == 422
