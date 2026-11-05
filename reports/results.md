# Results of the latest run

_Generated automatically by `credit_risk.pipeline` on 05 Nov 2026 17:17. Data: LendingClub accepted loans 2007-2018Q4._

## Population
| step | loans |
| --- | --- |
| total_rows | 2260668 |
| missing_key_fields | 0 |
| policy_code_not_1 | 0 |
| not_36_month_term | 650914 |
| issued_outside_window | 1020119 |
| not_individual_application | 239 |
| unresolved_outcome | 147 |
| eligible | 589249 |

## Splits
| split | loans | bad_loans | bad_rate | first_issue | last_issue |
| --- | --- | --- | --- | --- | --- |
| train | 143892 | 18281 | 0.127 | 2012-01 | 2013-12 |
| validation | 162570 | 22315 | 0.137 | 2014-01 | 2014-12 |
| test | 282787 | 42089 | 0.149 | 2015-01 | 2015-12 |

**Target:** 1 = loan charged off or defaulted at any point during its 36-month term; 0 = fully paid. Lifetime default flag, not a 12-month PD.

## Model comparison (test year, same loans for every model)
| model | loans | bad_rate | auc | auc_ci_low | auc_ci_high | gini | ks | brier | log_loss | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scorecard | 282787 | 0.149 | 0.665 | 0.662 | 0.668 | 0.330 | 0.238 | 0.122 | 0.400 | 1.022 |
| gbm | 282787 | 0.149 | 0.682 | 0.679 | 0.685 | 0.363 | 0.264 | 0.120 | 0.395 | 0.972 |
| benchmark | 282787 | 0.149 | 0.693 | 0.690 | 0.696 | 0.386 | 0.280 | 0.121 | 0.396 | 0.962 |

**Champion: gbm.** On the 2014 validation year, gradient boosting improved AUC by 0.020 (95% CI 0.018 to 0.022) and stayed calibrated (slope 0.97), which outweighs the loss of simplicity.

## Extra challengers (reported, not deployed)
_None trained in this run._

Status: xgboost: skipped: xgboost not installed; torch_mlp: skipped: torch not installed

## Risk bands (test year)
| band | loans | bad_loans | bad_rate | mean_predicted_pd | share_of_loans |
| --- | --- | --- | --- | --- | --- |
| A | 48486 | 2163 | 0.045 | 0.041 | 0.171 |
| B | 67714 | 6259 | 0.092 | 0.080 | 0.239 |
| C | 69811 | 10266 | 0.147 | 0.123 | 0.247 |
| D | 58162 | 12040 | 0.207 | 0.181 | 0.206 |
| E | 38614 | 11361 | 0.294 | 0.279 | 0.137 |

## Policy outcomes at the configured cut-offs (test year, historical funded loans only)
| outcome | loans | observed_bad_rate | share |
| --- | --- | --- | --- |
| approve | 147107 | 0.085 | 0.520 |
| refer | 111054 | 0.196 | 0.393 |
| decline | 24626 | 0.317 | 0.087 |

Estimated LGD 0.89 and EAD 0.58 of the original amount (from 18,281 charged-off training loans). Expected lifetime loss on auto-approved test loans: 3.71% of the amount lent.

## Monitoring replay
| quarter | loans | score_psi | score_status | most_shifted_feature | its_psi | mean_predicted_pd | observed_bad_rate | auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014Q1 | 34074 | 0.001 | stable | verification_status | 0.098 | 0.128 | 0.129 | 0.672 |
| 2014Q2 | 37881 | 0.001 | stable | verification_status | 0.178 | 0.129 | 0.135 | 0.676 |
| 2014Q3 | 40595 | 0.004 | stable | verification_status | 0.158 | 0.130 | 0.136 | 0.682 |
| 2014Q4 | 50020 | 0.017 | stable | verification_status | 0.146 | 0.137 | 0.145 | 0.678 |
| 2015Q1 | 56568 | 0.012 | stable | verification_status | 0.185 | 0.135 | 0.148 | 0.675 |
| 2015Q2 | 64222 | 0.010 | stable | verification_status | 0.196 | 0.134 | 0.154 | 0.678 |
| 2015Q3 | 73567 | 0.005 | stable | verification_status | 0.211 | 0.131 | 0.146 | 0.686 |
| 2015Q4 | 88430 | 0.005 | stable | verification_status | 0.252 | 0.129 | 0.148 | 0.685 |

## Top features by Information Value
| feature | iv | strength | status |
| --- | --- | --- | --- |
| fico_mid | 0.1238 | medium | selected |
| annual_inc | 0.09333 | weak | selected |
| tot_cur_bal | 0.05424 | weak | selected |
| total_rev_hi_lim | 0.04512 | weak | selected |
| acc_open_past_24mths | 0.04466 | weak | selected |
| loan_to_income | 0.0444 | weak | selected |
| mort_acc | 0.0415 | weak | selected |
| dti | 0.04063 | weak | selected |
| percent_bc_gt_75 | 0.03578 | weak | selected |
| bc_util | 0.03481 | weak | dropped: correlation 0.76 with percent_bc_gt_75 |
