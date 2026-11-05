"""The data preparation step in PySpark, for running the full file on a Spark cluster such as Databricks.

It does exactly what sql/01_clean.sql, 02_population.sql and 03_features.sql do in DuckDB (and
what data.load_population_pandas does in pandas): read every column as text, convert types, give
each loan the first population rule it fails, keep the eligible loans, add the target and the
derived features. tests/test_spark_matches_pandas.py checks the three versions agree.

Why Spark at all, when DuckDB already handles 2.2 million rows on a laptop? Because a lender's
application tables are far bigger and live in a data platform, not on a laptop. This module shows
the same rules running where the data would really be, and the heavy step (reading, cleaning and
filtering ~2.2m rows and ~150 columns) is done by Spark; only the much smaller eligible population
comes back to pandas for modelling.

Notes for Spark 4 (ANSI mode on by default): all conversions use try_cast / try_to_timestamp, which
return NULL for a bad value instead of failing the whole job, matching DuckDB's TRY_CAST.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data import EXCLUSION_ORDER
from .features import RAW_COLUMNS, RAW_NUMERIC, build_features

log = logging.getLogger(__name__)

TEXT_COLUMNS = ["grade", "sub_grade", "home_ownership", "verification_status", "loan_status", "purpose",
                "addr_state", "application_type"]


def get_spark(app_name: str = "credit-risk"):
    """The running Spark session (Databricks provides one), or a new local one."""
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (SparkSession.builder.appName(app_name)
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def read_raw(spark, path: str | Path):
    """Every column as text. multiLine handles the free-text fields that contain line breaks."""
    from pyspark.sql import functions as F

    raw = spark.read.csv(str(path), header=True, inferSchema=False, multiLine=True, escape='"')
    for col in RAW_COLUMNS:
        if col not in raw.columns:
            raw = raw.withColumn(col, F.lit(None).cast("string"))
    return raw.select(*RAW_COLUMNS)


def clean(raw):
    """Same conversions as sql/01_clean.sql. Rows without a numeric id (the file's footer) are dropped."""
    from pyspark.sql import functions as F

    def num(col):
        return F.expr(f"try_cast(trim(replace({col}, '%', '')) AS DOUBLE)").alias(col)

    def month(col, alias):
        return F.to_date(F.expr(f"try_to_timestamp(trim({col}), 'MMM-yyyy')")).alias(alias)

    cols = [F.expr("try_cast(trim(id) AS BIGINT)").alias("id")]
    cols += [num(c) for c in RAW_NUMERIC]
    cols += [
        F.expr(r"try_cast(regexp_extract(term, '(\\d+)', 1) AS INT)").alias("term_months"),
        month("issue_d", "issue_date"),
        month("earliest_cr_line", "earliest_cr_date"),
        F.expr(r"CASE WHEN trim(emp_length) LIKE '<%' THEN 0 "
               r"ELSE try_cast(regexp_extract(emp_length, '(\\d+)', 1) AS INT) END").alias("emp_length_years"),
    ]
    cols += [F.trim(F.col(c)).alias(c) for c in TEXT_COLUMNS]
    return raw.select(*cols).where(F.col("id").isNotNull())


def flag_population(clean_df, cfg: dict):
    """The first population rule each loan fails (NULL = eligible). Mirrors sql/02_population.sql."""
    from pyspark.sql import functions as F

    p = cfg["population"]
    resolved = list(p["good_statuses"]) + list(p["bad_statuses"])
    app_type = p["application_type"].upper()
    reason = (
        F.when(F.col("issue_date").isNull() | F.col("loan_status").isNull() | F.col("term_months").isNull(),
               "missing_key_fields")
        .when(F.coalesce(F.col("policy_code"), F.lit(1.0)) != 1, "policy_code_not_1")
        .when(F.col("term_months") != p["term_months"], "not_36_month_term")
        .when((F.col("issue_date") < F.lit(p["issue_start"]).cast("date"))
              | (F.col("issue_date") > F.lit(p["issue_end"]).cast("date")), "issued_outside_window")
        .when(F.upper(F.coalesce(F.col("application_type"), F.lit(app_type))) != app_type,
              "not_individual_application")
        .when(~F.col("loan_status").isin(resolved), "unresolved_outcome")
    )
    return clean_df.withColumn("exclusion_reason", reason)


def population_with_features(flagged, cfg: dict):
    """Eligible loans + target + derived features (sql/02 and sql/03)."""
    from pyspark.sql import functions as F

    bad = list(cfg["population"]["bad_statuses"])
    return (flagged.where(F.col("exclusion_reason").isNull()).drop("exclusion_reason")
            .withColumn("bad", F.col("loan_status").isin(bad).cast("int"))
            .withColumn("fico_mid", (F.col("fico_range_low") + F.col("fico_range_high")) / 2.0)
            .withColumn("credit_history_months",
                        (F.year("issue_date") - F.year("earliest_cr_date")) * 12
                        + (F.month("issue_date") - F.month("earliest_cr_date")))
            .withColumn("loan_to_income",
                        F.when(F.col("annual_inc") > 0, F.col("loan_amnt") / F.col("annual_inc"))))


def audit_table(flagged) -> pd.DataFrame:
    """How many loans each rule removed, in the same layout as the other two readers."""
    from pyspark.sql import functions as F

    rows = (flagged.groupBy(F.coalesce(F.col("exclusion_reason"), F.lit("eligible")).alias("step"))
            .count().collect())
    counts = {r["step"]: int(r["count"]) for r in rows}
    total = sum(counts.values())
    steps = ["total_rows"] + EXCLUSION_ORDER + ["eligible"]
    loans = [total] + [counts.get(s, 0) for s in EXCLUSION_ORDER] + [counts.get("eligible", 0)]
    return pd.DataFrame({"step": steps, "loans": loans})


def load_population_spark(path: str | Path, cfg: dict, spark=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same result as data.load_population_duckdb, with Spark doing the heavy lifting."""
    spark = spark or get_spark()
    # No .cache(): Databricks serverless compute doesn't support it. Spark reads the file twice
    # (once for the audit counts, once for the population), which costs a minute, not correctness.
    flagged = flag_population(clean(read_raw(spark, path)), cfg)
    audit = audit_table(flagged)
    log.info("Spark: %s rows read, %s eligible", f"{audit['loans'].iloc[0]:,}", f"{audit['loans'].iloc[-1]:,}")
    pop = population_with_features(flagged, cfg).orderBy("id").toPandas()
    pop["issue_date"] = pd.to_datetime(pop["issue_date"])
    pop["earliest_cr_date"] = pd.to_datetime(pop["earliest_cr_date"])
    for col in RAW_NUMERIC + ["term_months", "emp_length_years", "credit_history_months"]:
        pop[col] = pd.to_numeric(pop[col], errors="coerce").astype("float64")
    pop["id"] = pop["id"].astype("int64")
    pop["bad"] = pop["bad"].astype("int64")
    return build_features(pop), audit
