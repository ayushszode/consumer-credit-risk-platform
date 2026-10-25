"""The DuckDB/SQL path and the pandas path must build exactly the same population and features."""
import numpy as np
import pytest

from credit_risk.data import load_population_duckdb, load_population_pandas
from credit_risk.features import MODEL_FEATURES


def test_sql_and_pandas_agree(cfg):
    pytest.importorskip("duckdb")
    path = cfg["data"]["raw_file"]
    pd_pop, pd_audit = load_population_pandas(path, cfg)
    sql_pop, sql_audit = load_population_duckdb(path, cfg)
    assert pd_audit["loans"].tolist() == sql_audit["loans"].tolist()
    a = pd_pop.sort_values("id").reset_index(drop=True)
    b = sql_pop.sort_values("id").reset_index(drop=True)
    assert a["id"].astype("int64").tolist() == b["id"].astype("int64").tolist()
    assert (a["bad"].to_numpy() == b["bad"].to_numpy()).all()
    for col in MODEL_FEATURES:
        if a[col].dtype.kind in "fi":
            assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True), col
        else:
            assert a[col].astype(str).tolist() == b[col].astype(str).tolist(), col
