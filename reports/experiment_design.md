# Testing a new model before it replaces the old one

If a lender wanted to switch from one model to another, how would it check the new one is better
without risking too much money? This note covers the offline comparison I can do with historical
data, and how a live test would be designed.

## Offline comparison (done)

Both models score the same 2015 loans. The challenger's cut-off is set so it approves the same share
of loans as the champion, so the comparison is fair. `results/offline_policy_comparison.csv` then
shows:

- loans both models would approve;
- loans only the champion would approve, and their bad rate;
- loans only the challenger would approve, and their bad rate.

The interesting part is the two "only" groups. If the challenger's extra approvals default less than
the champion's extra approvals, it is picking better loans at the same approval rate.

**What it cannot show.** This is not an experiment. Both groups were approved by LendingClub, so we
learn nothing about applicants that one model would approve and LendingClub declined.

## Live test (designed, not run)

| Item | Plan |
| --- | --- |
| Design | New applicants randomly assigned: most to the current model, a small share (for example 10%) to the challenger |
| Main measure | Bad rate on approved loans, at the same approval rate |
| Early warning | Share of loans 30 or more days in arrears after 3 and 6 months, because lifetime outcomes take three years |
| Guardrails | Stop if the challenger's early arrears are clearly worse, or if complaints or declines for any group rise |
| Sample size | From `experiment.required_sample_size`; saved as `live_experiment_loans_per_arm_for_1pt_change` in `results/summary.json` |
| Fairness | Compare decline rates and outcomes for available groups in both arms |

The sample size is the number of approved loans needed in each arm to detect a 1 percentage point
rise in bad rate, with 5% significance and 80% power. For a consumer lender that is a lot of loans,
which is why live tests often run for months and rely on early arrears.

`experiment.simulate_random_split` runs the same comparison on historical loans with a random split.
Its output is labelled SIMULATED and should only be used to check the code, not as evidence.

## Findings from the full run

Offline comparison on 2015 loans, with the challenger (scorecard) approving the same number of loans
as the champion (boosting): 147,107 each, 52.0% of loans.

| Group | Loans | Observed bad rate |
| --- | --- | --- |
| Approved by the champion | 147,107 | 8.5% |
| Approved by the challenger | 147,107 | 9.1% |
| Approved by both | 124,073 | 7.7% |
| Only the champion | 23,034 | 12.8% |
| Only the challenger | 23,034 | 16.8% |

The two models agree on 84% of approvals; the loans only the champion approves default much less
often than those only the challenger approves (12.8% against 16.8%). Detecting a 1-point difference in
bad rate in a live test needs about 12,849 loans per arm. Offline evidence already favours the
champion, so a live test would mainly check that the gain holds on applicants LendingClub declined,
who are not in this data.
