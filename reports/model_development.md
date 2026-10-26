# Model development

This note explains how the data, features and models were built. The numbers from each run are written
to `reports/results.md` and the CSV files in `reports/results/`; this page explains how to read them.

## 1. Data

**Source.** LendingClub accepted loans, 2007 to 2018 Q4, from the Kaggle copy described in
[`../data/README.md`](../data/README.md).

**Cleaning** (`sql/01_clean.sql`, with the same rules in `data.clean_raw` for pandas):

- every column is read as text first, then converted with `TRY_CAST`, so one bad value does not stop
  the load;
- dates such as `Dec-2015` are parsed into real dates;
- percentages stored as text (`13.56%`) and terms (`36 months`) become numbers;
- employment length becomes years (`< 1 year` becomes 0, `10+ years` becomes 10);
- the summary rows at the bottom of the file, which have no numeric loan id, are dropped.

**Population** (`sql/02_population.sql`). Each loan gets at most one exclusion reason, checked in
this order:

| Order | Rule | Why |
| --- | --- | --- |
| 1 | Missing id, issue date, status, term, policy code, application type or state | Cannot be placed in the population |
| 2 | `policy_code` is not 1 | The data dictionary marks policy code 2 as new products that were not publicly available |
| 3 | Term is not 36 months | 60-month loans have a different risk profile and take longer to mature |
| 4 | Issued before 2012 or after 2015 | Earlier years are small; later loans had mostly not finished by the end of the data |
| 5 | Joint application | Two applicants' details, a different product |
| 6 | Status is not Fully Paid, Charged Off or Default | Outcome not known yet |

The count removed by each rule is saved to `results/population_audit.csv`. If two rules apply to the
same loan, it is counted under the first one only, so the counts add up to the total.

**Target.** `bad = 1` for Charged Off or Default, `0` for Fully Paid. This is whether the loan
defaulted at any point in its life, not within 12 months.

**Splits by issue date.**

| Split | Issue dates | Used for |
| --- | --- | --- |
| Train | Jan 2012 to Dec 2013 | Binning, WoE, coefficients, boosting fit, LGD and EAD |
| Validation | Jan 2014 to Dec 2014 | Choosing boosting settings |
| Test | Jan 2015 to Dec 2015 | Final comparison, policy and loss analysis. Looked at once. |

`time_split` checks that no loan id appears in more than one split.

## 2. Features

`features.feature_register()` lists every column in the raw file and puts it in one of four groups.
The full list is saved to `results/feature_register.csv`.

| Group | Examples | Rule |
| --- | --- | --- |
| Use | FICO range, DTI, income, revolving utilisation, enquiries, delinquencies | Known when the person applies |
| Benchmark only | `grade`, `sub_grade`, `int_rate`, `installment` | LendingClub's own risk view. Only in the benchmark model |
| Exclude: leakage | `total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`, `last_fico_range_*` | Only known after the loan was funded |
| Needs review | Free text and anything not covered by a rule | Not used until someone decides |

`assert_no_leakage` runs at the start of every pipeline run and stops it if a leakage field is in
the model feature list. There is also a test for this.

**Derived features** (calculated the same way in SQL and in Python):

- `fico_mid`: the middle of the FICO range at application;
- `credit_history_months`: months from the first credit account to the issue date;
- `loan_to_income`: loan amount divided by annual income (left empty if income is zero or missing).

In total there are 25 numeric and 3 categorical candidate features (`features.py`).

## 3. Scorecard

1. **Binning.** Numeric features start with up to 8 bins at quantiles of the training data.
   Neighbouring bins are merged if one has less than 5% of loans, or if their bad rates are within
   0.2 percentage points of each other. Missing values get their own bin. Categories with less than
   5% of loans are grouped as `OTHER`.
2. **Weight of Evidence.** For each bin, `WoE = ln(share of good loans / share of bad loans)`, with
   0.5 added to each count so an empty bin does not give an infinite value. A bin first seen
   after training gets WoE 0, which means "no information".
3. **Selection.** Features with Information Value below 0.02 are dropped. Going from the highest IV
   down, a feature is also dropped if its WoE values have a correlation above 0.70 with one already
   kept. The reason for every drop is saved in `results/information_value.csv`.
4. **Logistic regression** on the WoE values (scikit-learn, `C = 1.0`).
5. **Points.** `factor = 20 / ln(2)` and `offset = 600 - factor x ln(20)`. Each feature's points for a
   bin are `-(coefficient x WoE + intercept / n) x factor + offset / n`, rounded. A higher score means
   lower risk. The full points table is `results/points_table.csv`.
6. **Reason codes.** For each applicant, the features where they are furthest below the best
   possible points for that feature, largest gap first.

## 4. Challenger and benchmark

**Gradient boosting** uses scikit-learn's `HistGradientBoostingClassifier` on the same raw features,
with the categorical columns passed as categories. Three settings are tried (see `configs/base.yaml`),
each with early stopping on a slice of the training data. The one with the best 2014 AUC is kept.
All three are in `results/boosting_grid.csv`.

**Benchmark.** The same model again, also given `int_rate` and `sub_grade`. It is not a candidate for
use. It answers one question: how much of LendingClub's own view can an independent model reproduce?

## 5. Choosing the champion

The rule was set before the test year was scored:

> Use boosting only if its test AUC is at least 0.01 higher than the scorecard's, the 95% confidence
> interval for that difference (paired bootstrap, same loans) is above zero, and its calibration slope
> is between 0.8 and 1.2. Otherwise keep the scorecard.

The rule leans towards the scorecard on purpose. It is simpler to explain, the points can be
checked by hand, and reason codes come straight from it. A small, uncertain gain in AUC is not worth
losing that.

## Findings from the full run

<!-- Fill from results.md after the real run:
     - population size and how many loans each rule removed
     - bad rate in each split, and whether it moves between years
     - which features the scorecard kept and dropped, and the five with the highest IV
     - chosen boosting settings and number of iterations
     - test AUC / Gini / KS for all three models, and the champion decision -->
