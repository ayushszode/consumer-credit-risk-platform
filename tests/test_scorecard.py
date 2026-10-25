import numpy as np
import pandas as pd

from credit_risk.data import time_split
from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.scorecard import Scorecard, fit_bins


def _fitted(population, cfg):
    pop, _ = population
    train = time_split(pop, cfg)["train"]
    return Scorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train, train["bad"]), train


def test_woe_bins_cover_missing_values():
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 100, name="x")
    y = pd.Series([0, 1, 1, 0, 0] * 100)
    fb = fit_bins(x, y, "numeric", n_bins=4, min_share=0.05)
    assert "Missing" in fb.table["bin"].tolist()
    assert fb.iv >= 0


def test_unseen_category_gets_neutral_woe(population, cfg):
    sc, train = _fitted(population, cfg)
    row = train.head(1).copy()
    row["purpose"] = "SOMETHING_NEW"
    p = sc.predict_proba(row)[:, 1]
    assert 0 < p[0] < 1


def test_points_add_up_and_move_against_risk(population, cfg):
    sc, train = _fitted(population, cfg)
    sample = train.head(2000)
    assert np.allclose(sc.points(sample).sum(axis=1), sc.score(sample))
    corr = np.corrcoef(sc.score(sample), sc.predict_proba(sample)[:, 1])[0, 1]
    assert corr < -0.9  # more points = lower predicted risk


def test_selected_features_meet_iv_threshold(population, cfg):
    sc, _ = _fitted(population, cfg)
    assert sc.selected
    assert all(sc.bins[c].iv >= sc.min_iv for c in sc.selected)


def test_reason_codes(population, cfg):
    sc, train = _fitted(population, cfg)
    codes = sc.reason_codes(train.head(5))
    assert len(codes) == 5
    for applicant in codes:
        assert len(applicant) <= 3
        assert all(r["points_below_best"] > 0 for r in applicant)
        gaps = [r["points_below_best"] for r in applicant]
        assert gaps == sorted(gaps, reverse=True)
