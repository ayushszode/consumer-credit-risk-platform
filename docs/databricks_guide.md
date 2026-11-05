# Running the full project on Databricks

This runs everything on the real LendingClub file, with **PySpark** doing the data preparation and
**XGBoost** and **PyTorch** trained as extra challengers. Databricks **Free Edition** is enough.

You need: a Kaggle account (to download the data) and about 45 minutes, most of it waiting.

## 1. Download the data

1. Go to the Kaggle dataset **"Lending Club" by wordsforthewise**.
2. Download `accepted_2007_to_2018Q4.csv.gz` (about 375 MB). Keep it zipped.
3. Note the licence shown on the dataset page in `DATA_LICENSE_NOTICE.md`. The file itself never goes into Git.

## 2. Create a free Databricks account

1. Sign up for **Databricks Free Edition** with your email.
2. You land in a workspace with a catalog called `workspace` and a schema called `default`.

## 3. Create a volume and upload the file

1. Left menu: **Catalog** > `workspace` > `default` > **Create** > **Volume**.
2. Name it `lendingclub` (managed volume) and create it.
3. Open the volume > **Upload to this volume** > choose `accepted_2007_to_2018Q4.csv.gz`.

The file is now at `/Volumes/workspace/default/lendingclub/accepted_2007_to_2018Q4.csv.gz`.

## 4. Open this repository as a Git folder

1. Left menu: **Workspace** > your user folder > **Create** > **Git folder**.
2. Git repository URL: `https://github.com/ayushszode/consumer-credit-risk-platform`
3. Create. The whole project now appears in your workspace.

## 5. Run the notebook

1. Open `notebooks/databricks_full_run` inside the Git folder.
2. Connect to **Serverless** compute (top right).
3. **Run all**. The first cells install XGBoost and PyTorch and restart Python; the rest runs the project.
4. When it finishes, the last cell prints the name of a zip file saved in the volume.

If the PyTorch install fails (some workspaces block that download), run the notebook anyway: the
network is skipped and noted in the results, and everything else still runs.

## 6. Bring the results back

1. **Catalog** > `workspace` > `default` > `lendingclub` > the `credit_risk_results_....zip` file > **Download**.
2. Send the zip back. It holds `reports/results/` (every number), `reports/figures/`, `reports/results.md`
   and the trained model bundle. It contains no individual loan records.

Those files are what fill the README "Results" section, the reports, the LinkedIn post and the
portfolio case study. Nothing gets written there by hand.

## Running on your own computer instead

No Spark needed: DuckDB does the same data preparation on a laptop.

```bash
make setup
make setup-extras        # XGBoost, PyTorch (CPU) and PySpark
# put accepted_2007_to_2018Q4.csv.gz in data/raw/
make run                 # DuckDB data preparation + all models
make run-spark           # same, with PySpark doing the data preparation (needs Java 17+)
```

## What each part shows

| Part | What it shows |
|---|---|
| PySpark data preparation | The same population rules running where a lender's data would actually live; `tests/test_spark_matches_pandas.py` proves Spark, DuckDB and pandas give identical populations |
| XGBoost challenger | Whether a different gradient boosting library beats scikit-learn's on the same loans |
| PyTorch network | Whether a neural network adds anything over trees and a scorecard on tabular credit data |
| Champion rule | Unchanged: the extra challengers are reported, not deployed |
