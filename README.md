# Consumer Credit Risk Platform

[![tests](https://github.com/ayushszode/consumer-credit-risk-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ayushszode/consumer-credit-risk-platform/actions/workflows/ci.yml)

![Consumer Credit Risk Platform](images/cover.png)

**Live demo:** [ayushszode-credit-risk.streamlit.app](https://ayushszode-credit-risk.streamlit.app) (uses the model trained on the real LendingClub loans; the applicants you score there are made up)

![The app: score an applicant, see the decision and what drove it](images/app_demo.png)

End-to-end consumer credit risk project using **Python, SQL and scikit-learn** on public LendingClub
loan data. It goes from a raw loan file to a lending decision: an explainable scorecard, a challenger
model, approve / refer / decline cut-offs, expected loss, monitoring, and a small API and app to score
one applicant at a time. The data preparation can also run in **PySpark** (for example on Databricks), and **XGBoost**
and a **PyTorch** network can be tested as extra challengers on the same loans.

## Business objective

A lender needs to decide who to lend to, and it has to be able to explain each decision to the
customer, a credit committee and a regulator. This project asks:

1. Which information known at the time of application best separates loans that are repaid from
   loans that default?
2. Does a gradient boosting model rank applicants clearly better than a traditional points-based
   scorecard, or is the simpler model good enough?
3. If a lender had used the model on 36-month loans in 2015, how many could it have approved
   automatically, and how many of those would have defaulted?
4. How much should it have expected to lose on those loans?
5. How would the lender know if the model started to drift?

## Dataset

**LendingClub accepted loans, 2007 to 2018 Q4** (Kaggle: "All Lending Club loan data")

- Loans used: 36-month, individual applications issued **January 2012 to December 2015** that ended
  as Fully Paid, Charged Off or Default
- Target: `1` if the loan charged off or defaulted at any point in its term, `0` if fully paid
- Split by issue date: **train 2012 to 2013, validation 2014, test 2015**

Raw data is not committed to this repository. See [`data/README.md`](data/README.md) and
[`DATA_LICENSE_NOTICE.md`](DATA_LICENSE_NOTICE.md).

## Results

Full run on the real file (5 November 2026). Every number below is read from `reports/results/`; the
complete auto-generated summary is in [`reports/results.md`](reports/results.md).

**Population.** 2,260,668 loans in the file; **589,249 eligible** 36-month individual
loans issued 2012-2015 with a final outcome. The largest exclusions were loans issued outside the window
(1,020,119) and 60-month loans (650,914); only 147 loans were dropped for having no final outcome.

| Split | Issued | Loans | Bad rate |
| --- | --- | --- | --- |
| Train | 2012-2013 | 143,892 | 12.7% |
| Validation | 2014 | 162,570 | 13.7% |
| Test | 2015 | 282,787 | 14.9% |

**Model comparison on the 2015 test year** (the same 282,787 loans for every model):

| Model | AUC (95% CI) | Gini | KS | Calibration slope |
| --- | --- | --- | --- | --- |
| Scorecard (13 WoE features) | 0.665 (0.662-0.668) | 0.330 | 0.238 | 1.02 |
| Gradient boosting | 0.682 (0.679-0.685) | 0.363 | 0.264 | 0.97 |
| Benchmark incl. LendingClub grade and rate | 0.693 (0.690-0.696) | 0.386 | 0.280 | 0.96 |

- **Champion: gradient boosting.** Decided on the 2014 validation year
  with the rule set in advance: AUC +0.020 over the scorecard (95% CI 0.018 to 0.022), calibration slope 0.97. On 2015 the gain held at +0.017 (95% CI 0.015 to 0.018).
- The independent model comes within 0.012 AUC of a benchmark that also sees
  LendingClub's own grade and interest rate.
- **Calibration:** the ranking holds (slopes close to 1), but the models predict too low in 2015: mean predicted
  PD 13.2% against an observed 14.9%, because the bad rate rose from 12.7% in the training years. Before live use the intercept would be recalibrated;
  the monitoring replay below shows this gap opening quarter by quarter.

<p>
  <img src="reports/figures/roc_test.png" alt="ROC curves on the 2015 test year" width="49%">
  <img src="reports/figures/calibration_test.png" alt="Calibration on the 2015 test year" width="49%">
</p>

**Lending policy on 2015 loans** (cut-offs fixed in advance: approve below 12% PD, decline at 25% or above):

| Decision | Share of loans | Observed bad rate |
| --- | --- | --- |
| Approve | 52.0% | 8.5% |
| Refer | 39.3% | 19.6% |
| Decline | 8.7% | 31.7% |

- Risk bands A to E are in order: observed bad rate from 4.5% (A) to 29.4% (E).
- **Expected lifetime loss on auto-approved loans: 3.71% of the amount lent**, with
  LGD 0.89 and EAD 0.58 of the original amount, estimated from 18,281 charged-off training loans.

**Monitoring replay (2014 Q1 to 2015 Q4).** Score PSI stayed at or below 0.017 every quarter
(stable) and AUC stayed between 0.672 and 0.686. The input to investigate is
`verification_status` (PSI 0.20 between the training years and 2015).

**Extra challengers.** XGBoost and the PyTorch network were not part of this run (their libraries were not
installed where it ran). Their comparison is added by the GitHub Actions or Databricks run
(`docs/github_full_run.md`), in `reports/results/extra_challengers_vs_scorecard_test.csv`.

## Modelling approach

### Data preparation (SQL)

- DuckDB reads the compressed file directly and cleans every column with `TRY_CAST`, so one bad
  value does not stop the load
- Population rules are applied in a fixed order, and the number of loans removed by each rule is
  saved (`population_audit.csv`)
- A pandas version of the same rules runs if DuckDB is not installed, and a test checks both give
  the same result
- A **PySpark** version (`spark_data.py`) runs the same rules on a Spark cluster such as Databricks
  (`--engine spark`); a third test checks Spark gives exactly the same population as pandas

### Leakage control

Every column in the file is listed in a feature register as *use*, *benchmark only*, *exclude:
leakage* or *needs review*. Anything only known after a loan was funded (payments, recoveries, last
payment date, later credit scores) is blocked, and the pipeline stops if one gets into a model.

### Scorecard (main model)

- Weight-of-evidence binning on the training years only
- Features kept if Information Value is at least 0.02 and they are not highly correlated (above
  0.70) with a stronger feature
- Logistic regression, then points: **600 points at odds of 20 to 1, 20 points to double the odds**
- Reason codes come from where an applicant loses the most points

### Challenger and benchmark

- **Gradient boosting** (`HistGradientBoostingClassifier`) on the same features, tuned on 2014
- **Benchmark**: the same model also given LendingClub's own grade and interest rate, to show how
  close an independent model gets to the lender's own view. It is never a candidate for use
- **Extra challengers** (`challengers.py`), scored on the same 2015 loans and compared with the
  scorecard using a paired bootstrap:
  - **XGBoost**, small grid, early stopping on 2014
  - a small **PyTorch** network (two hidden layers, quantile-scaled inputs, missing-value flags,
    early stopping on 2014 log loss, no class weighting so the outputs stay usable as probabilities)
- The extra challengers are **reported, not deployed**: the champion rule below is unchanged. Both
  are optional and skipped automatically if the library isn't installed

### Champion rule (judged on 2014, so 2015 stays a clean test)

Boosting replaces the scorecard only if, on the 2014 validation year, it improves AUC by at least
**0.01**, the 95% confidence interval for the gain (paired bootstrap on the same loans) is above zero,
and its calibration slope stays between **0.8 and 1.2**. 2015 is used only to report the result.

### Lending policy and loss

```text
Risk bands:  A <= 6%   B <= 10%   C <= 15%   D <= 22%   E above 22%   (predicted PD)
Decision:    approve if PD <= 12%   decline if PD >= 25%   otherwise refer
Expected loss = PD x LGD x EAD
```

LGD and EAD are measured on charged-off training loans, with a sensitivity table for other values.
This is a lifetime expected loss, not an IFRS 9 calculation.

### Monitoring

Population Stability Index for the score and every feature (0.10 investigate, 0.25 act), a
quarter-by-quarter replay of 2014 and 2015, and default rates by vintage.

### Explanations

Each decision comes with a plain-English reason built from a template, for example:

> Your application has been declined. Our model puts it in risk band E. The main things that raised
> the estimated risk were a credit score of 677, 78% of available revolving credit in use and a
> debt-to-income ratio of 31%.

An optional language-model rewrite is off by default. If it changes a number, the outcome, or
mentions a protected topic, it is thrown away and the template is used.

## Repository structure

```text
consumer-credit-risk-platform/
├── README.md
├── START_HERE.md
├── DATA_LICENSE_NOTICE.md
├── LICENSE
├── Makefile
├── requirements.txt / requirements-dev.txt / requirements-extras.txt / constraints.txt
├── configs/
│   └── base.yaml                  # every threshold and setting in one place
├── sql/
│   ├── 01_clean.sql
│   ├── 02_population.sql
│   └── 03_features.sql
├── src/credit_risk/
│   ├── data.py  spark_data.py  features.py  quality.py
│   ├── scorecard.py  models.py  challengers.py  evaluate.py
│   ├── policy.py  pricing.py  monitoring.py  experiment.py
│   ├── explain.py  service.py  figures.py
│   ├── synthetic.py               # fake loans for tests and the demo
│   └── pipeline.py                # runs everything
├── api/
│   └── main.py                    # FastAPI: /health, /score
├── app/
│   └── streamlit_app.py           # 5-page interactive app (Plotly charts)
├── notebooks/
│   └── databricks_full_run.py     # full run on Databricks: Spark prep + all models
├── tests/                         # 52 tests, no real loan data needed
├── examples/
│   └── applicant.json
├── reports/
│   ├── model_development.md  validation_review.md  policy_analysis.md
│   ├── experiment_design.md  governance_mapping.md
│   ├── limitations.md  decision_log.md
│   └── results.md, results/, figures/   # generated by the pipeline
├── images/
│   └── cover.png
├── docs/
│   ├── case_study.md  interview_guide.md  data_dictionary.md
│   ├── linkedin_entry.md  cv_bullets.md  github_setup.md
│   ├── databricks_guide.md  github_full_run.md
├── data/
│   ├── README.md
│   ├── raw/                       # downloaded locally; excluded from Git
│   └── processed/                 # excluded from Git
└── artifacts/                     # trained model, generated locally
```

## How to run

```bash
conda create -n credit-risk python=3.12 -y
conda activate credit-risk
make setup        # install requirements and the package
make test         # 52 tests, synthetic data (5 skip without the extras below)
make demo         # whole pipeline on synthetic data, under a minute (overwrites reports/ and the model)
```

With the real file in `data/raw/`:

```bash
make sample       # 20% of the loans, for a quick check
make run          # everything
```

Then:

```bash
make api          # http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d @examples/applicant.json
make app          # Streamlit app in the browser
```

Optional extras (XGBoost, PyTorch CPU build, PySpark):

```bash
make setup-extras
make run-spark    # data preparation in PySpark (needs Java 17+ on your own machine)
```

To run everything on the real data without a laptop, either use **GitHub Actions**
([`docs/github_full_run.md`](docs/github_full_run.md): downloads the file from Kaggle and runs
Spark + all models on GitHub's servers) or **Databricks** (Free Edition is enough:
[`docs/databricks_guide.md`](docs/databricks_guide.md) and the notebook in `notebooks/`).

See [`START_HERE.md`](START_HERE.md) for the same steps one at a time.

## Reports

| Report | What it covers |
|---|---|
| [`model_development.md`](reports/model_development.md) | Data, population rules, features and models |
| [`validation_review.md`](reports/validation_review.md) | The model checked as a validator would, with open questions |
| [`policy_analysis.md`](reports/policy_analysis.md) | Bands, cut-offs, approval rates and expected loss |
| [`experiment_design.md`](reports/experiment_design.md) | How champion and challenger would be tested live |
| [`governance_mapping.md`](reports/governance_mapping.md) | PRA SS1/23, FCA Consumer Duty and IFRS 9 in plain terms |
| [`limitations.md`](reports/limitations.md) | What this project cannot tell you |
| [`decision_log.md`](reports/decision_log.md) | The main decisions and why |

## Limitations

- US online lender, 2012 to 2018. UK lenders use different bureau data, affordability rules and
  products, so the model itself would not transfer; the method would.
- Accepted loans only. Nobody LendingClub declined is in the data, so the model is likely too
  optimistic about applicants like them.
- The target is a lifetime default flag, not a 12-month PD, and expected loss is not IFRS 9.
- LGD and EAD are single averages, not models.
- Monitoring is a replay on past data, not live.
- This is a portfolio and learning project, not a production lending system.

## What I would do next

- Model *when* loans default (a survival model) to get a 12-month PD
- Replace the single LGD with a model that varies by loan
- Check outcomes across groups, carefully, using the proxies available
- Try the same pipeline on UK data if a suitable public dataset becomes available

## Skills demonstrated

**Credit Risk | Scorecards | Python | SQL | DuckDB | PySpark | Databricks | scikit-learn | XGBoost |
PyTorch | Logistic Regression | Gradient Boosting | Model Validation | Calibration | Monitoring (PSI) |
Expected Loss | FastAPI | Streamlit | Testing | GitHub Actions**

---

Ayush Sanjay Zode · [LinkedIn](https://www.linkedin.com/in/ayushsanjayzode) ·
[Portfolio](https://ayushszode-portfolio-nm22.vercel.app/) · [GitHub](https://github.com/ayushszode)
