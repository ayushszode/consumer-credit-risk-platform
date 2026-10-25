# Data Dictionary

Fields used by the models. The full list of every column in the raw file, and whether it may be
used, is generated as `reports/results/feature_register.csv`.

## Target

| Field | Meaning |
|---|---|
| `bad` | 1 if `loan_status` is Charged Off or Default; 0 if Fully Paid. Loans with any other status are excluded |

## Numeric features (25)

| Field | Meaning |
|---|---|
| `loan_amnt` | Loan amount requested |
| `annual_inc` | Annual income stated by the applicant |
| `loan_to_income` | Derived: loan amount / annual income |
| `dti` | Monthly debt payments / monthly income, %, excluding mortgage and the new loan |
| `fico_mid` | Derived: middle of the FICO range at application |
| `emp_length_years` | Years in current job (0 to 10, where 10 means 10 or more) |
| `credit_history_months` | Derived: months from first credit account to issue date |
| `inq_last_6mths` | Credit enquiries in the last 6 months |
| `mths_since_recent_inq` | Months since the most recent enquiry |
| `delinq_2yrs` | 30+ day late payments in the last 2 years |
| `mths_since_last_delinq` | Months since the last late payment (missing if none) |
| `pub_rec` | Derogatory public records |
| `pub_rec_bankruptcies` | Public record bankruptcies |
| `open_acc` | Open credit lines |
| `total_acc` | Total credit lines in the credit file |
| `acc_open_past_24mths` | Accounts opened in the last 24 months |
| `num_actv_rev_tl` | Active revolving accounts |
| `mort_acc` | Mortgage accounts |
| `revol_bal` | Total revolving balance |
| `revol_util` | Revolving credit used, % of limit |
| `bc_util` | Bank card credit used, % of limit |
| `percent_bc_gt_75` | % of bank cards over 75% of their limit |
| `total_rev_hi_lim` | Total revolving credit limit |
| `tot_cur_bal` | Total current balance on all accounts |
| `mo_sin_rcnt_tl` | Months since the most recent account was opened |

## Categorical features (3)

| Field | Values |
|---|---|
| `home_ownership` | RENT, MORTGAGE, OWN, OTHER |
| `verification_status` | Not Verified, Source Verified, Verified |
| `purpose` | debt_consolidation, credit_card, home_improvement, and others |

## Benchmark only

| Field | Meaning |
|---|---|
| `int_rate` | Interest rate LendingClub set |
| `sub_grade` | LendingClub's risk grade (A1 to G5) |

## Used only for loss estimates, never as model inputs

| Field | Meaning |
|---|---|
| `funded_amnt` | Amount actually lent |
| `total_rec_prncp` | Principal repaid |
| `recoveries` | Amount recovered after charge-off |
| `collection_recovery_fee` | Fee paid for collection |

Definitions follow the LendingClub data dictionary that comes with the Kaggle dataset.
