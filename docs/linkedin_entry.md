# LinkedIn Projects Entry

## Project name

**Consumer Credit Risk Platform**

## Description

Built an end-to-end consumer credit risk project on public LendingClub loan data, from raw file to lending decision. Prepared the data with SQL (DuckDB), with a matching PySpark version, blocked post-funding fields with a feature register and automated checks, and trained on 2012 to 2013, tuned on 2014 and tested on 2015.

Built an explainable weight-of-evidence scorecard and a gradient boosting challenger, choosing between them with a rule set in advance. On 589,249 loans, boosting reached a 2015 test AUC of 0.682 (scorecard 0.665). At a 12% PD cut-off, 52% of 2015 loans would be approved automatically with an 8.5% bad rate and 3.71% expected loss. Added risk bands, PSI monitoring and a FastAPI service.

The live interactive app lets you score your own applicant: it shows the decision, a risk dial, plain-English reasons and how the risk changes with the credit score, plus a cut-off slider for the lending policy. 52 automated tests run on GitHub Actions.

**Tools:** Python | SQL | DuckDB | PySpark | pandas | scikit-learn | FastAPI | Streamlit | Plotly | pytest | GitHub Actions

**Media:** the live app link, with `images/app_demo.png` as its thumbnail.

## Suggested skills

- Credit Risk
- Credit Scoring
- Risk Modelling
- Python
- SQL
- scikit-learn
- Machine Learning
- Model Validation
- Statistical Modelling
- Data Analysis

## Post

```
I've finished a credit risk project. The code is on GitHub, and there's a live app you can try.

I wanted to go the whole way from a raw loan file to a lending decision, not stop at a model score. Using public LendingClub data (589,249 personal loans over 36 months, issued 2012 to 2015), it:

- cleans the data with SQL and records every loan it removes and why
- builds a points-based scorecard and tests it against gradient boosting, using a rule I set before looking at the test year (boosting won: AUC 0.682 against 0.665 on 2015)
- turns predictions into approve, refer and decline cut-offs: at a 12% cut-off, 52% of loans are approved automatically with an 8.5% bad rate
- estimates expected loss (3.71% of the amount lent) and checks for drift quarter by quarter

The most useful finding was a problem. Defaults rose from 12.7% in the training years to 14.9% in 2015, so the model still ranked risk well but predicted it too low. Before real use it would need recalibrating, and the monitoring shows that gap opening quarter by quarter.

It's US data and only covers accepted loans, so it says nothing about people who were turned down, and it wouldn't transfer to UK lending as it is.

In the app you can score a made-up applicant and move the lending cut-off yourself. Happy to talk it through with anyone working in credit risk or pricing.

Code: https://github.com/ayushszode/consumer-credit-risk-platform
Live app: https://ayushszode-credit-risk.streamlit.app
```
