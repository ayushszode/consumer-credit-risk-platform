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
| 16 | Add a PySpark version of the data preparation, tested against pandas | Spark only; DuckDB only | Shows the same rules running on a data platform (Databricks) without losing the laptop path |
| 17 | XGBoost and a PyTorch network as extra challengers, reported not deployed | Replace the main challenger; skip them | Answers "did you try X?" with evidence, without changing the champion rule after the fact |
| 18 | No class weighting in the network | Weight defaults more heavily | Weighting distorts predicted probabilities, which the policy and expected loss depend on |
| 19 | Extra libraries optional, with a separate CI job | Make them required | The core project installs quickly; the extras are still tested on every push |

## Changes after the real run

| # | Date | Change | Why | Effect |
| --- | --- | --- | --- | --- |
| 20 | 2026-11-05 | Champion rule judged on the 2014 validation year instead of 2015 | 2015 should only report the result, never help choose it | None: boosting won on both years (+0.020 on 2014, +0.017 on 2015) |
| 21 | 2026-11-05 | Real run used the pandas reader | DuckDB and the extras were not installed where it ran | None on the population: the three readers are checked to agree |
