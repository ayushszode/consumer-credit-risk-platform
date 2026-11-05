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
- Set the champion rule in advance and judged it on 2014: boosting must improve AUC by at least 0.01,
  with a confidence interval above zero, and stay calibrated.
- Set risk bands and cut-offs, estimated expected loss as PD x LGD x EAD, and ran a quarterly
  monitoring replay with PSI.
- Wrapped the model in a FastAPI service and a Streamlit app, with 48 automated tests and GitHub
  Actions.

## Results

- 589,249 eligible loans from 2,260,668; bad rate 12.7% (2012-13), 13.7% (2014), 14.9% (2015)
- 2015 test AUC: scorecard 0.665, gradient boosting 0.682, benchmark with LendingClub's grade 0.693
- Champion: gradient boosting (+0.020 AUC on 2014, 95% CI 0.018 to 0.022; +0.017 on 2015)
- At the 12% cut-off, 52.0% of 2015 loans approved automatically, with an observed bad rate of 8.5%
- Expected lifetime loss on approved loans: 3.71% of the amount lent
- Score PSI at most 0.017 per quarter; `verification_status` is the input that shifted most (PSI 0.20)
- Main issue found: predicted PDs about 1.7 points too low in 2015, so the level needs recalibrating

## Business interpretation

Approving below a 12% PD would have let through about half of the 2015 loans automatically, with a
bad rate of 8.5% against 14.9% for all funded loans, and sent the riskiest 8.7% (31.7% bad rate) to
decline. Boosting earned its place: it passed the rule on 2014 and kept most of the gain on 2015
(+0.017 AUC), and when both models approve the same number of loans, the loans only boosting approves
go bad far less often than those only the scorecard approves (12.8% against 16.8%).

## Recommendation

Use gradient boosting to rank applications, keep the scorecard as the explainable reference and
fallback, and recalibrate the probability level on the latest complete year before the cut-offs drive
decisions, because defaults rose faster than either model expected. Watch `verification_status` and
small-business loans in monitoring.

## Limitations

US data from one online lender, accepted loans only, lifetime default rather than 12-month PD, and
single-average LGD and EAD. See [`../reports/limitations.md`](../reports/limitations.md).
