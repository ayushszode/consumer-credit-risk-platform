import numpy as np
import pandas as pd
import pytest

from credit_risk import evaluate, experiment, monitoring, policy, pricing

POLICY = {"version": "test", "bands": {"A": 0.06, "B": 0.10, "C": 0.15, "D": 0.22, "E": 1.0},
          "approve_max_pd": 0.12, "decline_min_pd": 0.25}


@pytest.mark.parametrize("pd_value,expected", [(0.05, "approve"), (0.12, "approve"), (0.1201, "refer"),
                                               (0.2499, "refer"), (0.25, "decline"), (0.9, "decline")])
def test_policy_boundaries(pd_value, expected):
    assert policy.decide(pd_value, POLICY) == expected


def test_risk_band_boundaries():
    assert policy.risk_band([0.0, 0.06, 0.0601, 0.15, 0.5], POLICY["bands"]).tolist() == ["A", "A", "B", "C", "E"]


def test_threshold_curve_is_monotonic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pd": rng.uniform(0, 0.5, 2000)})
    df["bad"] = (rng.random(2000) < df["pd"]).astype(int)
    curve = policy.threshold_curve(df, "pd")
    assert curve["selected_share"].is_monotonic_increasing


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 500), rng.random(500)
    m = evaluate.summary(y, s)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)


def test_psi_behaviour():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    assert monitoring.psi(a, rng.normal(size=5000)) < 0.02
    assert monitoring.psi(a, rng.normal(1, 1, size=5000)) > 0.25
    assert monitoring.status(0.05) == "stable" and monitoring.status(0.15) == "investigate"


def test_sample_size_grows_for_smaller_effects():
    assert experiment.required_sample_size(0.1, 0.005) > experiment.required_sample_size(0.1, 0.02)


def test_lgd_estimate_is_a_share():
    df = pd.DataFrame({"bad": [1, 1, 0], "funded_amnt": [1000.0, 2000.0, 500.0], "total_rec_prncp": [400.0, 500.0, 500.0],
                       "recoveries": [60.0, 0.0, 0.0], "collection_recovery_fee": [10.0, 0.0, 0.0]})
    est = pricing.estimate_lgd_ead(df)
    assert 0 <= est["lgd"] <= 1 and 0 <= est["ead_share"] <= 1
    assert est["lgd"] == pytest.approx(np.mean([1 - 50 / 600, 1.0]))
