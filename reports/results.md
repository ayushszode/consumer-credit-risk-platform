# Results of the latest run

_Generated automatically by `credit_risk.pipeline` on 05 Nov 2026 14:15. Data: synthetic demo data._

> **These results come from synthetic demo data. They only show the pipeline works; they say nothing about real credit risk.**

## Population
| step | loans |
| --- | --- |
| total_rows | 80000 |
| missing_key_fields | 0 |
| policy_code_not_1 | 0 |
| not_36_month_term | 19741 |
| issued_outside_window | 16477 |
| not_individual_application | 856 |
| unresolved_outcome | 284 |
| eligible | 42642 |

## Splits
| split | loans | bad_loans | bad_rate | first_issue | last_issue |
| --- | --- | --- | --- | --- | --- |
| train | 21501 | 3645 | 0.170 | 2012-01 | 2013-12 |
| validation | 10798 | 2029 | 0.188 | 2014-01 | 2014-12 |
| test | 10343 | 2027 | 0.196 | 2015-01 | 2015-12 |

**Target:** 1 = loan charged off or defaulted at any point during its 36-month term; 0 = fully paid. Lifetime default flag, not a 12-month PD.

## Model comparison (test year, same loans for every model)
| model | loans | bad_rate | auc | auc_ci_low | auc_ci_high | gini | ks | brier | log_loss | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scorecard | 10343 | 0.196 | 0.655 | 0.640 | 0.666 | 0.309 | 0.230 | 0.151 | 0.473 | 0.988 |
| gbm | 10343 | 0.196 | 0.654 | 0.640 | 0.667 | 0.307 | 0.225 | 0.151 | 0.474 | 0.953 |
| benchmark | 10343 | 0.196 | 0.648 | 0.635 | 0.659 | 0.296 | 0.219 | 0.151 | 0.475 | 0.864 |
| xgboost | 10343 | 0.196 | 0.658 | 0.644 | 0.670 | 0.316 | 0.229 | 0.150 | 0.472 | 1.014 |
| torch_mlp | 10343 | 0.196 | 0.654 | 0.640 | 0.665 | 0.307 | 0.239 | 0.150 | 0.472 | 0.972 |

**Champion: scorecard.** The scorecard was kept: boosting changed test AUC by -0.001 (95% CI -0.006 to 0.006; calibration slope 0.95). That gain is not large or certain enough to give up a model whose every point can be explained.

## Extra challengers (reported, not deployed)
| model | auc_change_vs_scorecard | ci_low | ci_high | calibration_slope |
| --- | --- | --- | --- | --- |
| xgboost | 0.004 | -0.002 | 0.009 | 1.014 |
| torch_mlp | -0.001 | -0.007 | 0.007 | 0.972 |

Status: xgboost: trained; torch_mlp: trained

## Risk bands (test year)
| band | loans | bad_loans | bad_rate | mean_predicted_pd | share_of_loans |
| --- | --- | --- | --- | --- | --- |
| A | 320 | 18 | 0.056 | 0.051 | 0.031 |
| B | 1783 | 175 | 0.098 | 0.082 | 0.172 |
| C | 2516 | 358 | 0.142 | 0.125 | 0.243 |
| D | 3020 | 626 | 0.207 | 0.183 | 0.292 |
| E | 2704 | 850 | 0.314 | 0.281 | 0.261 |

## Policy outcomes at the configured cut-offs (test year, historical funded loans only)
| outcome | loans | observed_bad_rate | share |
| --- | --- | --- | --- |
| approve | 3076 | 0.100 | 0.297 |
| refer | 5488 | 0.202 | 0.531 |
| decline | 1779 | 0.342 | 0.172 |

Estimated LGD 0.92 and EAD 0.61 of the original amount (from 3,645 charged-off training loans). Expected lifetime loss on auto-approved test loans: 4.95% of the amount lent.

## Monitoring replay
| quarter | loans | score_psi | score_status | most_shifted_feature | its_psi | mean_predicted_pd | observed_bad_rate | auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014Q1 | 2573 | 0.010 | stable | mths_since_last_delinq | 0.017 | 0.173 | 0.195 | 0.661 |
| 2014Q2 | 2724 | 0.002 | stable | mths_since_last_delinq | 0.011 | 0.171 | 0.175 | 0.686 |
| 2014Q3 | 2755 | 0.006 | stable | mths_since_last_delinq | 0.015 | 0.173 | 0.192 | 0.658 |
| 2014Q4 | 2746 | 0.004 | stable | mths_since_last_delinq | 0.012 | 0.170 | 0.190 | 0.677 |
| 2015Q1 | 2629 | 0.003 | stable | dti | 0.017 | 0.168 | 0.193 | 0.641 |
| 2015Q2 | 2755 | 0.017 | stable | dti | 0.026 | 0.176 | 0.193 | 0.654 |
| 2015Q3 | 2588 | 0.005 | stable | mths_since_last_delinq | 0.025 | 0.174 | 0.200 | 0.651 |
| 2015Q4 | 2371 | 0.007 | stable | dti | 0.022 | 0.174 | 0.198 | 0.675 |

## Top features by Information Value
| feature | iv | strength | status |
| --- | --- | --- | --- |
| fico_mid | 0.1739 | medium | selected |
| dti | 0.06542 | weak | selected |
| loan_to_income | 0.03009 | weak | selected |
| inq_last_6mths | 0.02514 | weak | selected |
| revol_util | 0.02194 | weak | selected |
| bc_util | 0.01998 | not useful | dropped: IV 0.01998 below 0.02 |
| annual_inc | 0.01761 | not useful | dropped: IV 0.01761 below 0.02 |
| loan_amnt | 0.01251 | not useful | dropped: IV 0.01251 below 0.02 |
| mths_since_recent_inq | 0.0107 | not useful | dropped: IV 0.0107 below 0.02 |
| credit_history_months | 0.009287 | not useful | dropped: IV 0.009287 below 0.02 |
