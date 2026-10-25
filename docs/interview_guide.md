# Interview Guide

Read the code alongside this. Every answer below should be something you can say in your own words.

## 30-second explanation

"I built a credit risk project on public LendingClub loan data that goes from the raw file to a
lending decision. I cleaned the data with SQL, made sure nothing known only after a loan was given
could get into the model, and trained on 2012 to 2013, tuned on 2014 and tested on 2015. The main
model is a points-based scorecard, and I tested it against gradient boosting using a rule I set
before looking at the test year. Then I turned the predictions into approve, refer and decline
cut-offs, estimated expected loss, and added monitoring and a small API that gives reasons for each
decision."

## Why split by year instead of randomly?

A model is always used on applicants who come after the ones it learned from. A random split mixes
2015 loans into training, which flatters the results. Testing on a later year shows whether the
model holds up when the population changes.

## What is leakage, and how did you stop it?

Leakage is when the model uses information it would not have at the time of the decision. In this
file, fields like total payments, recoveries and last payment date are only known after the loan
has run. I listed every column in a feature register, marked those as excluded, and added a check
that stops the pipeline if one is in the feature list.

## Why not use LendingClub's grade and interest rate?

They are LendingClub's own risk assessment. Using them would mostly copy LendingClub's model rather
than build an independent one. I kept them in a separate benchmark to see how close my model gets.

## What is weight of evidence and Information Value?

Each feature is split into bins. WoE for a bin is the log of (share of good loans in the bin /
share of bad loans in the bin), so positive means safer than average. IV adds up how different good
and bad loans look across the bins. Below 0.02 a feature is not useful, 0.02 to 0.1 is weak, 0.1 to
0.3 medium, above 0.3 strong.

## How do the points work?

The scorecard is a logistic regression on WoE values, rescaled so that 600 points means odds of 20
good loans to 1 bad, and every 20 extra points doubles those odds. Each applicant's score is the
sum of points across their bins, so you can see exactly where they gained or lost points.

## Why keep the scorecard if boosting might be better?

The rule was: boosting wins only if it improves AUC by at least 0.01, the confidence interval for
the improvement is above zero, and it stays calibrated. A small or uncertain gain is not worth giving
up a model that a customer, underwriter or regulator can follow point by point. If boosting had
passed the rule, I would have used it.

## Why does calibration matter, not just AUC?

AUC only says whether the model ranks people in the right order. Cut-offs and expected loss use the
actual probability, so if the model says 10% and the real rate is 15%, the policy and loss figures
will be wrong even if the ranking is fine. I checked calibration slope and a table of predicted vs
observed by group.

## What is PSI?

Population Stability Index compares how a variable is distributed now with how it was in training.
Below 0.10 is little change, 0.10 to 0.25 is worth investigating, above 0.25 is a large shift. I ran
it for the score and every feature, quarter by quarter.

## How did you calculate expected loss?

Expected loss = PD x LGD x EAD. PD comes from the model. LGD (share of the outstanding balance not
recovered) and EAD (balance at charge-off as a share of the loan) are averages from charged-off loans
in the training years. Because they are single averages, I also show a sensitivity table.

## Is this IFRS 9?

No. IFRS 9 needs a 12-month PD for stage 1 and lifetime for stages 2 and 3, discounting and
forward-looking scenarios. My target is lifetime default and there is no discounting. I would need a
model of when loans default (a survival model) to get there.

## What is the biggest limitation?

The data only has accepted loans. The model has never seen anyone LendingClub turned down, so it is
probably too optimistic about applicants like them. This is the reject inference problem, and I did
not try to fix it because there is no outcome data to check a fix against.

## How would you test a new model in a real lender?

Offline first: score the same historical loans with both models at the same approval rate and look
at the loans only one model would approve. Then a live test with a small random share of applicants
going to the new model, using early arrears at 3 and 6 months as the guardrail, because full
outcomes take three years.

## How does this relate to UK regulation?

PRA SS1/23 sets out model risk management principles for banks with internal model approval:
model identification, governance, development, independent validation and risk mitigants. The FCA
Consumer Duty is about good customer outcomes, including clear explanations and fair value. I mapped
the project against both in `reports/governance_mapping.md`, including what is missing, such as
independent validation and an affordability check.

## If asked "did you use AI?"

Be straightforward: you used an AI assistant to help write code and documentation, and you made
sure you understand and can explain every part, which is why you can answer these questions.
