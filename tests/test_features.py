import numpy as np
import pandas as pd
import pytest

from credit_risk.features import (LEAKAGE_FIELDS, LENDER_FIELDS, LOSS_FIELDS, MODEL_FEATURES, assert_no_leakage,
                                  build_features, feature_register, months_between)


def test_model_features_contain_no_post_origination_fields():
    assert not set(MODEL_FEATURES) & (LEAKAGE_FIELDS | set(LOSS_FIELDS))


def test_lender_fields_are_not_model_features():
    assert not set(MODEL_FEATURES) & set(LENDER_FIELDS)


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError):
        assert_no_leakage(MODEL_FEATURES + ["recoveries"])


def test_register_marks_leakage_and_unknown_columns():
    reg = feature_register(["loan_amnt", "total_pymnt", "grade", "some_new_column"]).set_index("field")
    assert reg.loc["loan_amnt", "decision"] == "use"
    assert reg.loc["total_pymnt", "decision"].startswith("exclude")
    assert reg.loc["grade", "decision"] == "benchmark only"
    assert reg.loc["some_new_column", "decision"] == "needs review"


def test_months_between():
    s = pd.Series(pd.to_datetime(["2010-01-01", "2014-12-01"]))
    e = pd.Series(pd.to_datetime(["2012-03-01", "2015-01-01"]))
    assert months_between(s, e).tolist() == [26, 1]


def test_build_features():
    df = pd.DataFrame({
        "loan_amnt": [10000.0, 5000.0], "annual_inc": [50000.0, 0.0], "fico_range_low": [700.0, 660.0],
        "fico_range_high": [704.0, 664.0], "earliest_cr_date": pd.to_datetime(["2005-06-01", "2010-01-01"]),
        "issue_date": pd.to_datetime(["2015-06-01", "2015-01-01"]), "home_ownership": ["rent", None],
        "verification_status": ["Verified", "Not Verified"], "purpose": ["car", "other"],
    })
    out = build_features(df)
    assert out["fico_mid"].tolist() == [702.0, 662.0]
    assert out["credit_history_months"].tolist() == [120, 60]
    assert out.loc[0, "loan_to_income"] == pytest.approx(0.2)
    assert np.isnan(out.loc[1, "loan_to_income"])  # zero income -> missing, not infinity
    assert out["home_ownership"].tolist() == ["RENT", "MISSING"]
