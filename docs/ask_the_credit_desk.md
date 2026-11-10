# Ask the Credit Desk: an AI assistant over the results

A small tool-using agent, in the app's **Ask the data** page, that answers plain-English questions
about this project ("What share of 2015 applicants would the policy approve?", "Score an applicant
with a credit score of 690...") by querying the published result tables and the live model.

The interesting part is not the chat box. It's the three things around it: **what the model is allowed
to touch, how we check what it says, and how we measure whether it's any good.**

## How it works

```text
question ──► language model (Gemini, function calling)
                 │  decides which tool to call, with what arguments
                 ▼
             tools ── list_tables       which tables exist, and their columns
                   ── run_sql           one read-only SELECT over the result tables
                   ── headline_metrics  the headline numbers from summary.json
                   ── score_applicant   the same scoring function the API uses
                 │  results go back to the model (up to 6 rounds)
                 ▼
             answer ──► grounding check ──► shown with every query it ran
```

Code: [`src/credit_risk/agent.py`](../src/credit_risk/agent.py). The model is reached through Gemini's
REST API with the Python standard library, so there is no extra dependency.

## What the model is allowed to touch

- **No loan records.** The agent only sees the aggregate result tables in `reports/results/` (bands,
  policy outcomes, metrics, segments, monitoring, data cleaning). 25 tables are listed explicitly;
  anything else in that folder is not loaded.
- **Read-only, enforced twice.** The SQL text is checked first (one statement, `SELECT`/`WITH` only,
  no write or admin keywords). Then the tables sit in an in-memory SQLite database with an
  *authorizer* that refuses anything except reading, so even a query that slipped past the text
  check could not change data. A test runs a `DELETE` straight against the connection to prove it.
- **Limits.** At most 50 rows come back, a runaway query is stopped by an operation budget, and the
  agent gets at most 6 rounds of tool calls.
- **Decisions stay with the policy engine.** `score_applicant` calls the same `score_applicant`
  function as the API; the model only reports what it returns.

## How we check what it says

Every number in the final answer must be traceable to a tool result (or to the question), allowing
for rounding and for fractions written as percentages (0.0850 → 8.5%). Small counts and years are
ignored. If a number can't be traced, the answer is still shown but flagged:
"⚠ these numbers were not found in any tool result". The app also shows every SQL query the agent ran.

This is a guardrail, not a guarantee: a number can be traced and still be used in a misleading sentence.

## How we measure it

[`evals/agent_questions.yaml`](../evals/agent_questions.yaml) holds 30 questions of five kinds:

| Type | Count | Passes when |
|---|---|---|
| number | 17 | the answer contains the right number |
| text | 8 | the answer names the right category (state, purpose, feature, model) |
| refuse | 2 | the tables can't answer it, and the agent says so without inventing numbers |
| safety | 2 | asked to delete or change data, it refuses and the table is unchanged |
| score | 1 | the risk band and decision match the model |

The correct answers are **never typed in**: each question has an `answer_sql` that computes the
answer from the same tables, so the test stays right if the results change.

```bash
GEMINI_API_KEY=... python -m credit_risk.agent_eval      # or: make agent-eval
```

This writes `reports/results/agent_eval.csv` (every question, answer, SQL and verdict) and
`reports/agent_evaluation.md` (the summary). The **Agent evaluation** GitHub Actions workflow runs the
same thing on demand using a repository secret.

The tests in [`tests/test_agent.py`](../tests/test_agent.py) don't need a key or a network: they replace
the model with a scripted stand-in and check the guardrails, the agent loop, the grounding check and
the grading code.

## Setting it up

1. Create a free API key in Google AI Studio (aistudio.google.com → *Get API key*).
2. **Never commit the key.** Put it in one of these places:
   - local app: `.streamlit/secrets.toml` (already in `.gitignore`):
     ```toml
     GEMINI_API_KEY = "your-key"
     ```
   - Streamlit Community Cloud: app → *Settings* → *Secrets*, same line as above.
   - evaluation workflow (the only place the public project needs it): GitHub repo → *Settings* →
     *Secrets and variables* → *Actions* → *New repository secret* named `GEMINI_API_KEY`.
3. Optional: `GEMINI_MODEL` (default `gemini-3.8-flash`) if you want a different model name from
   AI Studio's list.

**Live or demo mode.** With `GEMINI_API_KEY` in the app's secrets, visitors can type their own
questions (12 per visit, to protect the quota). Without it, the page runs in **demo mode**: it shows
the real answers, verdicts and SQL from the last evaluation run (`reports/results/agent_eval.csv`), so
nobody can use up the key. The public demo runs in demo mode.

The evaluation workflow starts when it is run from the Actions tab or when a branch named `eval/...`
is pushed, and commits its results back to that branch.

## Limitations

- It only knows what is in the result tables. Questions about individual borrowers, other years or
  other lenders should get "the tables don't cover that", and the evaluation checks this.
- Language models are not deterministic, even at temperature 0, so evaluation scores can move a
  little between runs.
- The grounding check catches invented numbers, not misleading wording.
