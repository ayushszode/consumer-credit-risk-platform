# CV Version

The exact wording used on the CV (numbers from `reports/results/` and `reports/agent_evaluation.md`).

**Consumer Credit Risk Platform | Python, SQL (DuckDB), scikit-learn, PySpark, FastAPI, Streamlit, Gemini API**

- Built an end-to-end credit decision workflow on public LendingClub loan data, using SQL to prepare a
  documented population of 589,249 loans and a feature register with automated checks to block
  post-funding (leakage) fields.
- Developed an explainable weight-of-evidence scorecard and a gradient boosting challenger, trained on
  2012-2013 and tested out of time on 2015; boosting won under a rule set in advance with an AUC of
  0.682 (scorecard 0.665).
- Turned predictions into risk bands and approve / refer / decline cut-offs: 52% of 2015 loans approved
  automatically with an 8.5% bad rate and 3.71% expected loss (PD x LGD x EAD); added PSI monitoring
  and a live Streamlit app and FastAPI service with reason codes, covered by 98 automated tests.
- Added an LLM assistant (Gemini function calling) that answers plain-English questions by writing
  read-only SQL and calling the scoring API, with database-level guardrails, a check that every number
  is traced to the data and a 30-question evaluation (30/30 correct).

XGBoost and the PyTorch network are not on the CV: they were not part of the real-data run.

For a shorter CV, keep bullets 2, 3 and 4. For AI or data science roles, put bullet 4 second.
