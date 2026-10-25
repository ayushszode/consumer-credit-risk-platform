# Case Study: Consumer Credit Risk Platform

## Situation

An unsecured personal lender has to decide quickly who to lend to, at what risk, and be able to
explain each decision. A model that only produces a score is not enough: the lender also needs
cut-offs, an estimate of losses, a way to compare a new model with the current one, and a way to
notice when the model stops working.

LendingClub's public loan data contains the application details and final outcome of over two
million US personal loans, which makes it possible to build and test that whole chain.

## Task

Build a credit decision workflow that:

1. prepares a clean, documented population with SQL and records every exclusion,
2. blocks any information that would not have been known at the time of application,
3. builds an explainable scorecard and tests it fairly against a gradient boosting model,
4. turns predicted risk into approve / refer / decline decisions with expected loss,
5. monitors stability over time,
6. serves decisions with reasons through an API and an app.

## Approach

- Selected 36-month individual loans issued 2012 to 2015, whose outcomes are known.
- Defined default as charged off or defaulted at any point in the loan's life.
- Split by issue date: trained on 2012 to 2013, tuned on 2014, tested once on 2015.
- Wrote a feature register for every column, and a test that fails if a post-funding field is used.
- Built a weight-of-evidence scorecard (600 points at 20:1 odds, 20 points to double the odds).
- Tuned a gradient boosting challenger on 2014 only, and built a benchmark that also sees
  LendingClub's own grade and rate.
- Set the champion rule before looking at 2015: boosting must improve AUC by at least 0.01, with a
  confidence interval above zero, and stay calibrated.
- Set risk bands and cut-offs, estimated expected loss as PD x LGD x EAD, and ran a quarterly
  monitoring replay with PSI.
- Wrapped the model in a FastAPI service and a Streamlit app, with 41 automated tests and GitHub
  Actions.

## Results

_To be added from the full run. Every figure here will be copied from `reports/results/`._

<!-- - population after exclusions; bad rate by year
     - test AUC / Gini / KS / calibration slope for scorecard, boosting and benchmark
     - champion and why
     - % of 2015 loans auto-approved and their observed bad rate
     - expected loss on approved loans
     - most shifted feature in monitoring -->

## Business interpretation

_To be written after the full run: what the cut-offs would mean for approval volume and losses, and
whether the extra complexity of boosting is worth it._

## Recommendation

_To be written after the full run._

## Limitations

US data from one online lender, accepted loans only, lifetime default rather than 12-month PD, and
single-average LGD and EAD. See [`../reports/limitations.md`](../reports/limitations.md).
