# LinkedIn Projects Entry

## Project name

**Consumer Credit Risk Platform**

## Description

Built an end-to-end consumer credit risk project on public LendingClub loan data, from raw file to
lending decision. Prepared the data with SQL (DuckDB), blocked post-funding fields with a feature
register and automated checks, and trained on 2012 to 2013, tuned on 2014 and tested on 2015.

Built an explainable weight-of-evidence scorecard and tested it against a gradient boosting
challenger using a champion rule set before scoring the test year. Turned predictions into risk
bands and approve / refer / decline cut-offs with expected loss (PD x LGD x EAD), added PSI
monitoring, and served decisions with plain-English reasons through a FastAPI service and a
Streamlit app, with automated tests on GitHub Actions.

**Tools:** Python | SQL | DuckDB | pandas | scikit-learn | FastAPI | Streamlit | pytest | GitHub Actions

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

## Post (no numbers until the full run)

```
I've built a credit risk project and put the code on GitHub.

I wanted to go the whole way from a raw loan file to a lending decision, not stop at a model score.
Using public LendingClub data (36-month personal loans issued 2012 to 2015), it:

- cleans the data with SQL and records every loan it removes and why
- builds a points-based scorecard and tests it against a gradient boosting model, using a rule I set
  before looking at the test year
- turns predictions into approve, refer and decline cut-offs, with an expected loss estimate
- checks for drift over time, and serves decisions through a small API and app

The parts I learned most from were the unglamorous ones: making sure nothing known only after a loan
was given leaked into the model, and testing on a later year instead of a random split.

I've also written up the limits. It's US data and only covers accepted loans, so it says nothing
about people who were turned down, and it wouldn't transfer to UK lending as it is.

Results from the full run are coming next. Happy to talk it through with anyone working in
credit risk or pricing.

https://github.com/ayushszode/consumer-credit-risk-platform
```
