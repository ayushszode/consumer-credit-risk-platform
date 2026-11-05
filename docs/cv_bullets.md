# CV Version

**Consumer Credit Risk Platform | Python, SQL, PySpark, scikit-learn, XGBoost, PyTorch, FastAPI**

- Built an end-to-end credit decision workflow on public LendingClub loan data, using DuckDB SQL to
  prepare a documented population and a feature register with automated checks to block
  post-funding (leakage) fields.
- Developed an explainable weight-of-evidence scorecard and a gradient boosting challenger on
  589,249 loans; boosting won under a rule set in advance and reached an out-of-time AUC of 0.682
  (scorecard 0.665).
- Translated predictions into risk bands and approve / refer / decline cut-offs: 52% of 2015 loans
  approved automatically with an 8.5% bad rate and 3.71% expected loss (PD x LGD x EAD); added PSI
  monitoring and a FastAPI / Streamlit service with reason codes, covered by 52 automated tests.
- (Only after the Databricks run is done:) Ran the data preparation in PySpark on Databricks (same population as the DuckDB and pandas
  versions, checked by a test) and compared XGBoost and a PyTorch neural network with the scorecard
  on the same out-of-time loans.

After the full run, add one verified result to the second or third bullet (for example the test AUC
or the approval rate and bad rate at the cut-off), copied exactly from `reports/results/`.

Use these bullets only once the GitHub project is uploaded and you can confidently explain the
methodology in interview.
