# Ask the Credit Desk: evaluation

Model: `gemini-3.5-flash-lite`. Questions: 30. Expected answers are computed from the result tables by SQL (see `evals/agent_questions.yaml`).

- **Correct:** 30 of 30 (100%)
- **Grounded** (every number traced to a tool result): 30 of 30
- **Average tool calls per question:** 1.5
- **Queries blocked by the guardrails:** 1
- **Median time per question:** 4.5 s

| Question type | Correct |
|---|---|
| number | 17 / 17 |
| refuse | 2 / 2 |
| safety | 2 / 2 |
| score | 1 / 1 |
| text | 8 / 8 |

## Questions it got wrong

None.
