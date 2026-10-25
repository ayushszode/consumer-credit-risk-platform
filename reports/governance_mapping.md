# How this lines up with UK expectations

This is a learning project on US data, so none of the rules below apply to it. I wrote this note to
show I understand what a UK lender would need to do before using a model like this, and where the
project falls short. It is not legal or compliance advice.

## PRA SS1/23: model risk management principles for banks

SS1/23 sets out the PRA's expectations for how banks manage model risk. It applies to UK banks,
building societies and PRA-designated investment firms that have approval to use internal models for
regulatory capital. The first version took effect in May 2024 and the PRA updated it in April 2026;
check the [Bank of England page](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss)
for the current text. It has five principles.

| Principle | What it asks for, briefly | What this project has | What is missing |
| --- | --- | --- | --- |
| 1. Model identification and model risk classification | Know which models you have and how risky each one is | Model version in every API response; scope and intended use in `validation_review.md` | No model inventory or formal risk tier |
| 2. Governance | Clear ownership, sign-off and reporting to senior management | `decision_log.md` records decisions and reasons | No independent owner, approver or committee |
| 3. Model development, implementation and use | Sound data, documented methods, testing, and use within known limits | `model_development.md`, feature register, leakage checks, out-of-time test, 41 automated tests, `limitations.md` | Not tested on UK data or a real decision system |
| 4. Independent model validation | Review by people who did not build the model | `validation_review.md` uses the same questions | It is a self-review, so it is not independent |
| 5. Model risk mitigants | Controls when a model is weak or untested, such as overlays or limits | Champion rule, refer band, PSI thresholds, template fallback for explanations | No formal overlay or use restrictions |

## FCA Consumer Duty

The Consumer Duty asks firms to deliver good outcomes for retail customers. Its four outcomes, and
how a credit model touches each one:

| Outcome | Where a credit model matters | In this project |
| --- | --- | --- |
| Products and services | Lending to people who can repay | Risk bands and cut-offs. A real lender also needs a separate affordability check, which this does not do |
| Price and value | Price reflects risk and gives fair value | Illustrative risk-based rate by band (`policy_analysis.md`) |
| Consumer understanding | Customers understand decisions | Plain-English reasons for each decision; the language-model rewrite is checked and falls back to a template |
| Consumer support | Customers can ask questions or challenge a decision | Refer band sends borderline cases to a person; the API returns reason codes that staff can explain |

Consumer Duty also expects firms to look at outcomes for different groups of customers, including
those in vulnerable circumstances. The data here has no information on vulnerability or protected
characteristics, which is listed in `limitations.md`.

## Other points a UK lender would add

- **Affordability.** FCA rules (CONC) require a creditworthiness assessment that covers whether the
  borrower can repay without hardship, not only whether they are likely to default.
- **Data protection.** Automated decisions with significant effects on people come with specific
  UK GDPR rules, including giving meaningful information about the logic involved.
- **IFRS 9.** Provisions need 12-month and lifetime expected credit loss with staging. This project's
  expected loss is a simpler lifetime figure (see `policy_analysis.md`).
- **Data.** A UK model would use UK credit reference agency data (for example Experian, Equifax or
  TransUnion) rather than US FICO scores.

## Why the model would not transfer to the UK

The population, credit bureau data, products, interest rates and economic conditions are all
different. What does transfer is the method: the population rules, leakage checks, out-of-time
testing, champion rule, monitoring and documentation.
