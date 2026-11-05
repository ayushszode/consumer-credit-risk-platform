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
| Data | Is every exclusion counted and explained? | `results/population_audit.csv` | Pass: 2,260,668 rows to 589,249, every loan in one step |
| Data | Are there duplicate loans, impossible values or dates out of order? | `results/data_quality.csv` | Pass, with a note: 0 duplicate ids; 11 loans with `revol_util` above 150% kept and flagged |
| Data | Does missingness change over time (a sign the data collection changed)? | `results/missing_by_year.csv` | Watch: `mths_since_recent_inq` missing 25% in 2012 against about 10% later; `emp_length` 4% to 7% |
| Leakage | Could any input only be known after funding? | `results/feature_register.csv`, `test_features.py` | Pass: register + test |
| Leakage | Do SQL and Python give the same population? | `test_sql_matches_python.py` | Pass (test, run in CI) |
| Implementation | Does the PySpark version give the same population as pandas? | `test_spark_matches_pandas.py` | Pass (test, run in the CI extras job) |
| Discrimination | Does the model rank good and bad loans apart on a year it has not seen? | AUC, Gini, KS in `results/metrics.csv` | Pass: 2015 AUC 0.682 (boosting), 0.665 (scorecard) |
| Discrimination | Is the difference between models real or noise? | Paired bootstrap in `results/summary.json` | Pass: paired interval above zero in 2014 and 2015 |
| Discrimination | Do XGBoost or the PyTorch network add anything over the scorecard? | `results/extra_challengers_vs_scorecard_test.csv` | Open: not yet run on the real data |
| Calibration | Do predicted PDs match observed bad rates? | `results/calibration_test_*.csv`, `figures/calibration_test.png` | Issue: slope about 1, but 2015 mean predicted 13.2% against 14.9% observed |
| Calibration | Are the risk bands in the right order with no overlap? | `results/bands_test.csv` | Pass: 4.5% (A) to 29.4% (E) |
| Stability | Does performance hold quarter by quarter? | `results/monitoring_replay.csv` | Pass: AUC 0.672 to 0.686, score PSI at most 0.017 |
| Stability | Which features shift the most between training and test years? | `results/drift_test_vs_train.csv` | Watch: `verification_status` PSI 0.20 |
| Segments | Is the model much weaker for any home-ownership type, loan purpose or state? | `results/segments_*.csv` | Watch: small business (AUC 0.637) and vacation (0.629) loans; renters 0.666 against 0.684 for mortgage holders |
| Conceptual soundness | Do the WoE patterns make sense (for example, risk falls as FICO rises)? | `results/points_table.csv` | Pass: FICO points rise from 36 to 58 |
| Explainability | Do reason codes match the points and read clearly? | `test_service_and_explain.py`, app page 1 | Pass (tests) |
| Implementation | Does the API path give the same PD as the pipeline for the same loan? | `test_service_matches_pipeline_for_the_same_loans` | Pass (test) |
| Implementation | Does the API reject impossible input? | `test_pipeline_and_api.py` | Pass (test) |

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
4. **Why boosting rather than the scorecard?** It passed the rule set in advance on 2014 (+0.020 AUC,
   calibration slope 0.97) and held its gain on 2015 (+0.017). Its reason codes come from swapping
   each feature to a typical value; the scorecard stays as the explainable reference and fallback.
5. **What happens to applicants like the ones LendingClub rejected?** Unknown. This is the largest
   open limitation (see `limitations.md`).
6. **Are WoE bins stable?** Bins are fitted on two years of data with at least 5% of loans each.
   A useful next check would be to refit on 2012 only and 2013 only and compare.
7. **Is the calibration good enough to set cut-offs on?** Look at the calibration slope and the
   band table. If the model is consistently above or below the observed rate, the cut-offs should be
   set on observed band bad rates, not raw PDs.

## Open findings

| # | Finding | Severity | Action |
| --- | --- | --- | --- |
| 1 | Predicted PDs are about 1.7 points too low in 2015 because the bad rate rose year by year | Medium | Recalibrate the intercept on the latest complete year before setting live cut-offs; monitor mean PD against observed rate each quarter |
| 2 | `verification_status` shifted between training and 2015 (PSI 0.20) | Low | Find out whether LendingClub changed how it verifies income; it is not in the scorecard, but boosting can use it |
| 3 | Weaker ranking for small-business and vacation loans | Low | Report by purpose in monitoring; consider a separate policy for small-business loans |
| 4 | XGBoost and the PyTorch network not yet compared on the real data | Low | Run the GitHub Actions or Databricks full run |
| 5 | No reject inference: only funded loans are in the data | Medium | State it in every use of the model (see `limitations.md`) |

## Overall view

The champion ranks risk consistently across 2014 and 2015 and its risk bands are in the right order,
so it is fit to rank applications for this portfolio. Before its cut-offs drove real decisions, the
probability level would need recalibrating to the latest year, because defaults rose faster than the
model expects, and the limits on rejected applicants would need to be accepted by the business.
