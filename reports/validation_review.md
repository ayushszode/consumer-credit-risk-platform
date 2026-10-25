# Validation-style review

I wrote this as if I were reviewing someone else's model. I am not an independent validator, since I
built the model myself, so treat this as a self-review against the questions a validation team
would ask.

## Scope

| Item | Detail |
| --- | --- |
| Model | Application scorecard for 36-month unsecured personal loans, with a gradient boosting challenger |
| Intended use | Rank and approve or decline new applications (in a real lender, together with affordability and fraud checks) |
| Output | Probability that the loan charges off or defaults during its term (lifetime, not 12 months) |
| Data | LendingClub accepted loans issued 2012 to 2015 |
| Not intended for | Regulatory capital, IFRS 9 provisions, UK lending, or any real lending decision |

## Checks and where to find them

| Area | Question | Evidence | Status |
| --- | --- | --- | --- |
| Data | Is every exclusion counted and explained? | `results/population_audit.csv` | _fill in_ |
| Data | Are there duplicate loans, impossible values or dates out of order? | `results/data_quality.csv` | _fill in_ |
| Data | Does missingness change over time (a sign the data collection changed)? | `results/missing_by_year.csv` | _fill in_ |
| Leakage | Could any input only be known after funding? | `results/feature_register.csv`, `test_features.py` | _fill in_ |
| Leakage | Do SQL and Python give the same population? | `test_sql_matches_python.py` | _fill in_ |
| Discrimination | Does the model rank good and bad loans apart on a year it has not seen? | AUC, Gini, KS in `results/metrics.csv` | _fill in_ |
| Discrimination | Is the difference between models real or noise? | Paired bootstrap in `results/summary.json` | _fill in_ |
| Calibration | Do predicted PDs match observed bad rates? | `results/calibration_test_*.csv`, `figures/calibration_test.png` | _fill in_ |
| Calibration | Are the risk bands in the right order with no overlap? | `results/bands_test.csv` | _fill in_ |
| Stability | Does performance hold quarter by quarter? | `results/monitoring_replay.csv` | _fill in_ |
| Stability | Which features shift the most between training and test years? | `results/drift_test_vs_train.csv` | _fill in_ |
| Segments | Is the model much weaker for any home-ownership type, loan purpose or state? | `results/segments_*.csv` | _fill in_ |
| Conceptual soundness | Do the WoE patterns make sense (for example, risk falls as FICO rises)? | `results/points_table.csv` | _fill in_ |
| Explainability | Do reason codes match the points and read clearly? | `test_service_and_explain.py`, app page 1 | _fill in_ |
| Implementation | Does the API path give the same PD as the pipeline for the same loan? | `test_service_matches_pipeline_for_the_same_loans` | _fill in_ |
| Implementation | Does the API reject impossible input? | `test_pipeline_and_api.py` | _fill in_ |

## Questions I would ask the developer

These are the challenges I would raise, with my answer as the developer. Some do not have a good
answer yet, and those are listed as open.

1. **Why a lifetime target?** The file only has final status, not monthly payments, so a 12-month
   default flag cannot be built reliably. The model should not be described as a 12-month PD.
2. **Why drop unresolved loans?** Their outcome is unknown. Because the window ends in 2015, the
   number dropped should be small; check the audit. If it were large, the remaining loans would lean
   towards ones that ended early, which includes early defaults.
3. **Is the test year representative?** 2015 was a period of fast growth for LendingClub, and its
   applicants may differ from 2012 and 2013. The drift report and the quarterly replay are there to
   show how much.
4. **Why is the scorecard champion when boosting might rank slightly better?** See the champion rule
   in `model_development.md`. If boosting had passed the rule, it would have been chosen.
5. **What happens to applicants like the ones LendingClub rejected?** Unknown. This is the largest
   open limitation (see `limitations.md`).
6. **Are WoE bins stable?** Bins are fitted on two years of data with at least 5% of loans each.
   A useful next check would be to refit on 2012 only and 2013 only and compare.
7. **Is the calibration good enough to set cut-offs on?** Look at the calibration slope and the
   band table. If the model is consistently above or below the observed rate, the cut-offs should be
   set on observed band bad rates, not raw PDs.

## Open findings

<!-- After the real run, list anything that did not pass, with a severity:
     High: would stop use. Medium: fix before use. Low: note and monitor.
     Example format:
     | # | Finding | Severity | Action |
     | 1 | ... | Medium | ... | -->

## Overall view

_To be written after the real run, in two or three sentences: is the model fit for the use described
above, and what would need to happen first._
