# Decision log

The main decisions, roughly in the order they were made. Each one says what the alternatives were
and why I chose as I did. Anything changed after the real run should be added at the bottom with a
date, rather than editing the earlier entries.

| # | Decision | Alternatives | Reason |
| --- | --- | --- | --- |
| 1 | Use the LendingClub accepted-loans file | Home Credit, Give Me Some Credit, a UK dataset | Large, public, has application data and final outcomes. No suitable public UK dataset |
| 2 | 36-month loans issued 2012 to 2015 | All years; include 60-month loans | Outcomes known for almost all of them; one product |
| 3 | Lifetime default target | 12-month default | No monthly payment history to build a 12-month flag |
| 4 | Drop unresolved loans and record how many | Treat them as good | Calling a running loan "good" would understate risk |
| 5 | Split by issue date (2012-13 / 2014 / 2015) | Random split | A model is used on later applicants, so it should be tested that way |
| 6 | Keep LendingClub's grade and rate out of the main model | Use them as features | They are LendingClub's own risk model; using them would mostly copy it |
| 7 | Write a feature register and block leakage fields in code | Choose columns by hand | A test catches mistakes; a reviewer can read the reasons |
| 8 | WoE scorecard as the main model | Plain logistic regression; boosting only | Easy to explain, standard in retail credit, reason codes come straight from the points |
| 9 | Gradient boosting as challenger, small grid | Bigger search; neural networks | Fair comparison without spending the test year |
| 10 | Champion rule set before scoring the test year | Pick the best test AUC | Avoids choosing a model after seeing the answer |
| 11 | Bands and cut-offs on predicted PD | On score points | PD is easier to link to bad rates and loss |
| 12 | LGD and EAD as training-year averages | LGD model | Enough for a first expected-loss figure; sensitivity table shows the effect |
| 13 | DuckDB SQL for data preparation, pandas as a fallback | pandas only | The full file is too large for pandas on a laptop with little memory |
| 14 | Template explanations, optional language model with checks | Language model only | Template is predictable; the model can only rephrase it |
| 15 | Synthetic data for tests and CI | Commit a sample of real data | Real loan records stay out of the repository |

## Changes after the real run

<!-- Add here with a date, for example:
     | 16 | 2026-11-02 | ... | ... | ... | -->
