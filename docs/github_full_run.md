# Full run on the real data with GitHub Actions

GitHub's own servers download the LendingClub file from Kaggle and run the whole project, with
PySpark doing the data preparation and every model trained, including XGBoost and PyTorch. Nothing
runs on your laptop and no Databricks account is needed. It takes about 30 to 60 minutes.

## One-off setup (about 3 minutes)

1. **Kaggle API key.** Sign in to kaggle.com > your profile picture > **Settings** > **API** >
   **Create New Token**.
   - If a file called `kaggle.json` downloads, open it: it holds a `username` and a `key`.
   - If instead a single token is shown, copy it.
2. **Add it to GitHub as secrets** (only you can see them; they never appear in the code or logs).
   Repository > **Settings** > **Secrets and variables** > **Actions** > **New repository secret**:
   - from `kaggle.json`: add `KAGGLE_USERNAME` (the username) and `KAGGLE_KEY` (the key), or
   - from a single token: add `KAGGLE_API_TOKEN`.

## Run it

Repository > **Actions** > **full run on the real data** > **Run workflow**.

When it finishes, open the run and download **credit-risk-results** at the bottom of the page. It
holds `reports/results/` (every number), `reports/figures/`, `reports/results.md` and the trained
model bundle. It contains no individual loan records, and the raw file is never saved anywhere
outside that one run.

## What the run does

1. Downloads the Kaggle dataset and keeps only `accepted_2007_to_2018Q4.csv.gz`
2. `python -m credit_risk.pipeline --engine spark`: PySpark reads and filters all ~2.2 million loans,
   then the scorecard, gradient boosting, benchmark, XGBoost and PyTorch models are trained and
   compared on the same 2015 loans
3. Prints the headline numbers in the log and uploads the results

The same run is also possible on Databricks (`docs/databricks_guide.md`) or on your own computer
(`make run`).
