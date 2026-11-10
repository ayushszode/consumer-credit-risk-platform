# Ask the Credit Desk: evaluation

Model: `gemini-3.5-flash-lite`. Questions: 30. Expected answers are computed from the result tables by SQL (see `evals/agent_questions.yaml`).

- **Correct:** 29 of 30 (97%)
- **Grounded** (every number traced to a tool result): 30 of 30
- **Average tool calls per question:** 1.7
- **Queries blocked by the guardrails:** 0
- **Median time per question:** 4.4 s

| Question type | Correct |
|---|---|
| number | 17 / 17 |
| refuse | 2 / 2 |
| safety | 2 / 2 |
| score | 0 / 1 |
| text | 8 / 8 |

## Questions it got wrong

**q30 (score)** Score an applicant with a credit score of 690, annual income of 42,000 dollars, a loan of 15,000 dollars and a debt-to-income ratio of 32%. Which risk band and what decision?  
Reason: expected band C and 'refer'  
Answer: The risk band is C and the policy outcome is refer. This result was returned by the live scoring tool for an applicant with a FICO score of 690, annual income of 42000, loan of 15000, and debt-to-income ratio of 32%.

