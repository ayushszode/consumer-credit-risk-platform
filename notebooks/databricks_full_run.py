# Databricks notebook source
# MAGIC %md
# MAGIC # Consumer credit risk platform: full run on the real data (Databricks)
# MAGIC
# MAGIC This notebook runs the whole project on the real LendingClub file:
# MAGIC
# MAGIC 1. **PySpark** reads and cleans all ~2.2 million loans and applies the population rules
# MAGIC 2. the **scorecard**, the **gradient boosting** challenger and the lender-grade **benchmark** are trained
# MAGIC 3. the two **extra challengers**, **XGBoost** and a small **PyTorch** network, are scored on the same 2015 loans
# MAGIC 4. policy, expected loss and monitoring tables are produced, and everything is zipped into one file to download
# MAGIC
# MAGIC **Before you start** (step-by-step instructions are in `docs/databricks_guide.md`):
# MAGIC - this repository is open in Databricks as a **Git folder**
# MAGIC - `accepted_2007_to_2018Q4.csv.gz` is uploaded to the volume `workspace.default.lendingclub`
# MAGIC
# MAGIC Run the cells from top to bottom. The full run takes roughly 15 to 30 minutes on Free Edition serverless compute.

# COMMAND ----------

# MAGIC %pip install xgboost pyyaml

# COMMAND ----------

# MAGIC %pip install torch --index-url https://download.pytorch.org/whl/cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC Change `RAW_FILE` only if you uploaded the file somewhere else.

# COMMAND ----------

import os
import sys

RAW_FILE = "/Volumes/workspace/default/lendingclub/accepted_2007_to_2018Q4.csv.gz"
VOLUME = "/Volumes/workspace/default/lendingclub"
RUN_DIR = "/tmp/credit_risk_run"  # local disk on the compute; the zip is copied to the volume at the end


def find_repo_root() -> str:
    """The Git folder this notebook lives in (one level above notebooks/)."""
    here = os.getcwd()
    for candidate in (here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(candidate, "src", "credit_risk")):
            return candidate
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return "/Workspace" + nb.rsplit("/notebooks/", 1)[0]


REPO = find_repo_root()
sys.path.insert(0, os.path.join(REPO, "src"))
print("Repository:", REPO)
print("Raw file found:", os.path.exists(RAW_FILE))

# COMMAND ----------

import yaml

from credit_risk.config import load_config

cfg = load_config(os.path.join(REPO, "configs", "base.yaml"))
cfg["data"].update(raw_file=RAW_FILE, processed_dir=f"{RUN_DIR}/processed",
                   artifacts_dir=f"{RUN_DIR}/artifacts", reports_dir=f"{RUN_DIR}/reports")
CONFIG_PATH = "/tmp/credit_risk_databricks.yaml"
with open(CONFIG_PATH, "w") as f:
    yaml.safe_dump(cfg, f)

from credit_risk import challengers  # noqa: E402

print("XGBoost installed:", challengers.is_available("xgboost"))
print("PyTorch installed:", challengers.is_available("torch"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick look at the raw file in Spark
# MAGIC Optional, but useful to see Spark doing the heavy step: how many loans per year and term, before any filtering.

# COMMAND ----------

from pyspark.sql import functions as F

from credit_risk.spark_data import clean, read_raw

raw_clean = clean(read_raw(spark, RAW_FILE))
display(raw_clean.groupBy(F.year("issue_date").alias("issue_year"), "term_months")
        .count().orderBy("issue_year", "term_months"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full run
# MAGIC `engine="spark"`: Spark prepares the population; the eligible loans then go to pandas and scikit-learn for modelling.

# COMMAND ----------

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

from credit_risk.pipeline import run  # noqa: E402

summary = run(CONFIG_PATH, engine="spark")
print(json.dumps({k: summary[k] for k in ("run_label", "engine", "champion", "champion_reason", "runtime_seconds")},
                 indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

import pandas as pd

results = f"{RUN_DIR}/reports/results"
display(pd.read_csv(f"{results}/population_audit.csv"))
display(pd.read_csv(f"{results}/metrics.csv").query("split == 'test'")
        [["model", "loans", "bad_rate", "auc", "auc_ci_low", "auc_ci_high", "gini", "ks", "brier", "calibration_slope"]])
display(pd.read_csv(f"{results}/extra_challengers.csv"))
display(pd.read_csv(f"{results}/extra_challengers_vs_scorecard_test.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save everything as one zip in the volume
# MAGIC Download it from **Catalog > workspace > default > Volumes > lendingclub** and send it back, so the README,
# MAGIC reports, LinkedIn post and portfolio can be filled with the real numbers.

# COMMAND ----------

import shutil
from datetime import datetime

stamp = datetime.now().strftime("%Y%m%d-%H%M")
local_zip = shutil.make_archive(f"/tmp/credit_risk_results_{stamp}", "zip", RUN_DIR, ".")
target = f"{VOLUME}/credit_risk_results_{stamp}.zip"
shutil.copy(local_zip, target)
print("Saved:", target, f"({os.path.getsize(target) / 1e6:.1f} MB)")
