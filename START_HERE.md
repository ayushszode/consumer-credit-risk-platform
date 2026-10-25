# Start Here

## 1. Create an environment and install

```bash
conda create -n credit-risk python=3.12 -y
conda activate credit-risk
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

(`make setup` does the last two lines.)

## 2. Check everything works

```bash
pytest
```

You should see 41 passed. The tests only use synthetic loans, so no download is needed.

## 3. Try the pipeline on synthetic data

```bash
python -m credit_risk.pipeline --synthetic
```

This takes under a minute and fills `reports/results.md`, `reports/results/` and `reports/figures/`.
The numbers are meaningless, but it shows every step running.

## 4. Download the real data

Follow [`data/README.md`](data/README.md). The file goes in `data/raw/`.

## 5. Run on the real data

```bash
python -m credit_risk.pipeline --sample 0.2   # quick check on 20% of loans
python -m credit_risk.pipeline                 # full run
```

## 6. Look at the results

- `reports/results.md`: every table from the run
- `reports/figures/`: ROC, calibration, policy trade-off, vintages, monitoring, Information Value
- `reports/validation_review.md`, `reports/policy_analysis.md`: what the numbers mean

## 7. Try the API and the app

```bash
uvicorn api.main:app --reload            # then open http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py
```

## 8. Before an interview

Read [`docs/interview_guide.md`](docs/interview_guide.md) and be ready to explain the champion rule,
the leakage checks and the limitations in your own words.
