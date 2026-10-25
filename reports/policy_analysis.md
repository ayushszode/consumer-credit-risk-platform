# Lending policy analysis

A model gives a number between 0 and 1. A lender needs a decision. This note covers how the
predicted probability of default (PD) becomes approve, refer or decline, and what that would have
meant for the 2015 loans.

## Risk bands

Bands are set on predicted PD (upper limits, in `configs/base.yaml`):

| Band | Predicted PD |
| --- | --- |
| A | up to 6% |
| B | over 6%, up to 10% |
| C | over 10%, up to 15% |
| D | over 15%, up to 22% |
| E | over 22% |

A band works if its observed bad rate sits close to its predicted PD and each band is riskier than
the one before. The check is in `results/bands_test.csv`.

## Cut-offs

| Outcome | Rule | What it would mean in practice |
| --- | --- | --- |
| Approve | PD 12% or lower | Automatic approval, subject to affordability and fraud checks |
| Refer | PD between 12% and 25% | Sent to an underwriter or asked for more information |
| Decline | PD 25% or higher | Automatic decline, with reasons given to the applicant |

These numbers are a starting point, not a recommendation. The threshold table
(`results/threshold_curve_test.csv`, chart in `figures/policy_tradeoff.png`) shows the share approved
and the bad rate for every cut-off from 4% to 40%, so a different risk appetite can be read off
directly.

## An important caveat

Every loan in this data was approved by LendingClub. When the analysis says "approve 30% of loans"
it means 30% of loans LendingClub already funded, not 30% of everyone who applied. The bad rates are
therefore for an already filtered group, and the people LendingClub declined are not in the picture
at all.

## Expected loss

For the loans the policy would approve automatically:

`expected loss = PD x LGD x EAD`

- **PD** is the model's prediction for each loan.
- **LGD** (loss given default) is the share of the outstanding balance not recovered. It is the
  average over charged-off loans in the training years: `1 - net recoveries / outstanding balance`.
- **EAD** (exposure at default) is the outstanding balance at charge-off as a share of the amount
  lent, again averaged over charged-off training loans.

Both use fields that are only known after a loan ends (`total_rec_prncp`, `recoveries`). That is fine
for estimating loss assumptions, and those fields are never model inputs.

Because LGD and EAD are single averages, `results/el_sensitivity.csv` shows expected loss for LGD
from 70% to 100% and EAD from 50% to 70%.

This is a lifetime expected loss for each loan. It is not an IFRS 9 expected credit loss, which would
need 12-month and lifetime stages, discounting and economic scenarios.

## Illustrative pricing

`results/rate_comparison.csv` compares, by band, LendingClub's average interest rate with a simple
rate that would cover expected loss plus funding cost (4%), operating cost (2%) and a margin (3%).
It is there to show how risk feeds into price, not as a pricing model. A real one would allow for
repayments over the life of the loan, early settlement, competitor rates and capital.

## Findings from the full run

<!-- Fill from results.md:
     - band table: are the bands in order, and how close are predicted and observed rates?
     - share approved / referred / declined at 12% and 25%, and the observed bad rate in each
     - expected loss on approved loans, with the LGD and EAD used
     - one sentence on what moving the approve cut-off from 12% to 10% or 15% would do
     - how the illustrative rate compares with LendingClub's rate by band -->
