"""Extra challengers (XGBoost, PyTorch). Skipped when the library isn't installed;
the 'extras' job in GitHub Actions installs both and runs these."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk import challengers
from credit_risk.data import time_split
from credit_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture(scope="module")
def parts(population, cfg):
    pop, _ = population
    return time_split(pop, cfg)


def _check_probabilities(model, df):
    p = model.predict_proba(df)
    assert p.shape == (len(df), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p[:, 1] > 0) & (p[:, 1] < 1)).all()
    return p[:, 1]


def test_xgboost_learns_the_signal(parts):
    pytest.importorskip("xgboost")
    m = challengers.XGBoostChallenger(NUMERIC_FEATURES, CATEGORICAL_FEATURES, n_estimators=400,
                                      early_stopping_rounds=25).fit(parts["train"], parts["validation"])
    p = _check_probabilities(m, parts["test"])
    assert roc_auc_score(parts["test"]["bad"], p) > 0.6
    assert 1 <= m.iterations_used <= 400


def test_xgboost_tuning_picks_best_validation_auc(parts):
    pytest.importorskip("xgboost")
    grid = [{"max_depth": 2, "min_child_weight": 20}, {"max_depth": 4, "min_child_weight": 50}]
    best, table = challengers.tune_xgboost(parts["train"], parts["validation"], NUMERIC_FEATURES,
                                           CATEGORICAL_FEATURES, grid, n_estimators=200, early_stopping_rounds=20)
    assert len(table) == 2
    best_auc = roc_auc_score(parts["validation"]["bad"], best.predict_proba(parts["validation"])[:, 1])
    assert np.isclose(best_auc, table["validation_auc"].max())


def test_torch_network_learns_and_is_repeatable(parts):
    pytest.importorskip("torch")
    kwargs = dict(hidden=(16, 8), max_epochs=8, patience=2, batch_size=256, random_state=3)
    a = challengers.TorchMLPChallenger(NUMERIC_FEATURES, CATEGORICAL_FEATURES, **kwargs).fit(parts["train"], parts["validation"])
    b = challengers.TorchMLPChallenger(NUMERIC_FEATURES, CATEGORICAL_FEATURES, **kwargs).fit(parts["train"], parts["validation"])
    pa = _check_probabilities(a, parts["test"])
    assert roc_auc_score(parts["test"]["bad"], pa) > 0.6
    assert np.allclose(pa, b.predict_proba(parts["test"])[:, 1], atol=1e-6)  # same seed, same answer
    assert 1 <= a.best_epoch_ <= len(a.history_) <= 8


def test_network_copes_with_missing_and_unseen_values(parts):
    pytest.importorskip("torch")
    m = challengers.TorchMLPChallenger(NUMERIC_FEATURES, CATEGORICAL_FEATURES, hidden=(8,), max_epochs=2,
                                       batch_size=512).fit(parts["train"], parts["validation"])
    odd = parts["test"].head(5).copy()
    odd[NUMERIC_FEATURES] = np.nan
    odd["purpose"] = "SOMETHING_NEW"
    _check_probabilities(m, odd)


def test_missing_library_is_skipped_not_fatal(parts, monkeypatch):
    monkeypatch.setattr(challengers, "is_available", lambda lib: False)
    settings = {"xgboost": {"enabled": True}, "torch_mlp": {"enabled": True}}
    models, status, tables = challengers.fit_extra_challengers(parts["train"], parts["validation"], NUMERIC_FEATURES,
                                                               CATEGORICAL_FEATURES, settings)
    assert models == {} and tables == {}
    assert status["status"].str.startswith("skipped").all()


def test_disabled_challenger_is_not_trained(parts):
    models, status, _ = challengers.fit_extra_challengers(parts["train"], parts["validation"], NUMERIC_FEATURES,
                                                          CATEGORICAL_FEATURES, {"xgboost": {"enabled": False}})
    assert models == {}
    assert set(status["status"]) == {"disabled in config"}
