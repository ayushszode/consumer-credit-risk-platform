# Limitations

What this project can and cannot tell you. I would read this before the results.

## Data

- **US lender, 2012 to 2018.** LendingClub was a US online lender. Its customers, products and the
  economic conditions of those years are different from today's UK market.
- **Accepted loans only.** Nobody LendingClub declined is in the data. The model has never seen
  those applicants, so it is probably too optimistic about people who look like them. I did not try
  reject inference because there is no outcome data to check it against.
- **Third-party copy.** The file is a Kaggle upload of data LendingClub used to publish, not an
  official release. Column meanings come from the data dictionary that came with it.
- **No protected characteristics.** The data has no age, sex, ethnicity or disability information, so
  I cannot check whether the model treats these groups fairly. `addr_state` and other fields could
  act as proxies for some of them.
- **Self-reported fields.** Income and employment length are partly self-reported, and only some
  incomes were verified.

## Target and population

- **Lifetime, not 12 months.** The outcome is whether a loan defaulted at any point in its 36 months.
  It is not the 12-month PD used for capital or IFRS 9 stage 1.
- **Dropped loans.** Loans still running at the end of the data are left out. The number is in the
  population audit.
- **36-month loans only.** 60-month loans were left out and would need their own model.

## Models

- **Single LGD and EAD.** One average each, from charged-off training loans. In reality these vary
  with the loan size, the borrower and the economy.
- **No economic variables.** The model does not know about unemployment or interest rates, so it
  cannot adjust to a downturn on its own.
- **Small boosting grid.** Three settings were tried. More tuning might help the challenger, but the
  point was a fair comparison, not the highest AUC.

## Policy and loss

- **Cut-offs are illustrations.** 12% and 25% are reasonable starting points. A real lender would set
  them from its risk appetite, capital and profit targets.
- **Not IFRS 9.** No staging, discounting or forward-looking scenarios.
- **Illustrative price only.** Ignores repayments over time, early settlement, competition and capital.

## Monitoring and experiment

- **Replay, not live.** The monitoring uses past quarters. It shows the method, not live behaviour.
- **Live test not run.** The experiment design is a plan.

## Explanations

- The template explanation is built from the model's reason codes, so it is only as good as those.
- The optional language-model rewrite is off by default. When on, it receives only the decision,
  risk band, PD and up to three reasons with the applicant's value for each, never a full loan
  record. If it changes a number, mentions a protected topic, changes the outcome or runs too long,
  it is discarded and the template is used.

## Use

This is a personal learning project. It must not be used for real lending decisions.
