"""The PySpark path must build exactly the same population and features as the pandas path.
Skipped when pyspark isn't installed; the 'extras' job in GitHub Actions runs it (with Java)."""
import numpy as np
import pytest

from credit_risk.data import load_population_pandas
from credit_risk.features import MODEL_FEATURES


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    session = (SparkSession.builder.master("local[2]").appName("credit-risk-tests")
               .config("spark.sql.session.timeZone", "UTC").config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    yield session
    session.stop()


def test_spark_and_pandas_agree(cfg, spark):
    from credit_risk.spark_data import load_population_spark
    path = cfg["data"]["raw_file"]
    pd_pop, pd_audit = load_population_pandas(path, cfg)
    sp_pop, sp_audit = load_population_spark(path, cfg, spark=spark)
    assert pd_audit["step"].tolist() == sp_audit["step"].tolist()
    assert pd_audit["loans"].tolist() == sp_audit["loans"].tolist()
    a = pd_pop.sort_values("id").reset_index(drop=True)
    b = sp_pop.sort_values("id").reset_index(drop=True)
    assert a["id"].astype("int64").tolist() == b["id"].astype("int64").tolist()
    assert (a["bad"].to_numpy() == b["bad"].to_numpy()).all()
    assert (a["issue_date"].to_numpy() == b["issue_date"].to_numpy()).all()
    for col in MODEL_FEATURES:
        if a[col].dtype.kind in "fi":
            assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True), col
        else:
            assert a[col].astype(str).tolist() == b[col].astype(str).tolist(), col
