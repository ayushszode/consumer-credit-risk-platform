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

## Why did boosting become the champion?

The rule, set in advance, was: boosting wins only if it improves AUC by at least 0.01 on the 2014
validation year, the confidence interval for the improvement is above zero, and it stays calibrated.
It passed: +0.020 AUC (95% CI 0.018 to 0.022), calibration slope 0.97. On 2015, which played no part
in the choice, the gain held at +0.017. So I used it, even though the rule leaned towards the
scorecard.

## How do you explain a boosting decision?

For each applicant I swap one feature at a time to a typical value and see how much the PD falls; the
three biggest falls are the reason codes. The scorecard is still kept: its points table shows the
pattern behind the model (for example, FICO points rising from 36 to 58), and it is the fallback if
the boosting model ever had to be withdrawn.

## What did the real run show?

589,249 loans; 2015 AUC 0.682 for boosting and 0.665 for the scorecard, against 0.693 for a model that
also sees LendingClub's own grade. At the 12% cut-off, 52% of 2015 loans would be approved
automatically with an 8.5% bad rate, and expected loss on them is 3.71% of the amount lent. The main
issue is that defaults rose from 12.7% to 14.9% across the years, so predicted PDs ran about 1.7
points low in 2015: the ranking is fine, the level needs recalibrating.

## Why Spark, when DuckDB already handles the file on a laptop?

DuckDB is the right tool for 2.2 million rows on one machine. Spark is what a lender's data platform
runs on, where application tables are far bigger and spread across many machines. I wrote the same
population rules in PySpark and ran them on Databricks, and a test checks that Spark, DuckDB and
pandas produce exactly the same loans and features. So the rules don't depend on the tool.

The heavy step, reading and cleaning about 150 columns for every loan, happens in Spark. Only the
eligible population, a few hundred thousand rows, comes back to pandas for the models.

## Did you try XGBoost or a neural network?

Yes, as extra challengers. Both are trained on 2012 to 2013, stopped early on 2014 and scored on the
same 2015 loans as the scorecard, and the AUC difference against the scorecard has a paired bootstrap
confidence interval. On tabular credit data, trees and well-built scorecards are usually hard to beat,
so the interesting result is how much (if anything) the extra complexity buys. (The real-data
comparison comes from the GitHub Actions or Databricks run; until then, say that it is set up and
tested, not that it has been run on the real data.)

## How did you set up the neural network?

A small network: two hidden layers (64 and 32 units), ReLU, a little dropout, trained with binary
cross-entropy and the Adam optimiser. Neural networks need inputs on similar scales and no gaps, so
numeric features are filled with the training median, passed through a quantile transform (which
handles skewed income and balances without hand-picked caps), and each feature that had gaps gets a
0/1 "was missing" flag. Categories are one-hot encoded. I didn't weight the classes, because that
would distort the probabilities, and probabilities are what the policy and expected loss use.

## What is early stopping, and doesn't it reuse the validation year?

After each pass over the training loans, I check the loss on 2014 and keep the best version; training
stops when it stops improving. It does use 2014 to choose, the same way the boosting grid does. That
is why 2015 is kept completely untouched: every number I quote comes from 2015.

## Why not deploy XGBoost or the network if it wins?

The champion rule was set before the test year was scored and covers the scorecard and the main
boosting model. If an extra challenger clearly won, the right step would be to record it in the
validation review and test it properly (calibration, explanations, stability by quarter), not to swap
it in quietly. In a bank, a new model type also means new validation and governance work.

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
