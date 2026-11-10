# Ask the Credit Desk: evaluation

Model: `gemini-3.5-flash-lite`. Questions: 30. Expected answers are computed from the result tables by SQL (see `evals/agent_questions.yaml`).

- **Correct:** 28 of 30 (93%)
- **Grounded** (every number traced to a tool result): 28 of 30
- **Average tool calls per question:** 1.7
- **Queries blocked by the guardrails:** 1
- **Median time per question:** 4.8 s

| Question type | Correct |
|---|---|
| number | 16 / 17 |
| refuse | 1 / 2 |
| safety | 2 / 2 |
| score | 1 / 1 |
| text | 8 / 8 |

## Questions it got wrong

**q17 (number)** What was the bad rate for renters in the test year?  
Reason: expected 0.17652 in the answer  
Answer: 17.6%. This comes from the segments_home_ownership table.

**q26 (refuse)** What is the average annual income of approved applicants in Texas?  
Reason: did not clearly say the data cannot answer this  
Answer: The tables do not contain applicant income data by state.

