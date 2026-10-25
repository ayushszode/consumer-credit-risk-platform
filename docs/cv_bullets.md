# CV Version

**Consumer Credit Risk Platform | Python, SQL, scikit-learn, FastAPI**

- Built an end-to-end credit decision workflow on public LendingClub loan data, using DuckDB SQL to
  prepare a documented population and a feature register with automated checks to block
  post-funding (leakage) fields.
- Developed an explainable weight-of-evidence scorecard and tested it against a gradient boosting
  challenger on an out-of-time year, using a champion rule set in advance (AUC gain, confidence
  interval and calibration).
- Translated predictions into risk bands, approve / refer / decline cut-offs and expected loss
  (PD x LGD x EAD), with PSI monitoring and a FastAPI / Streamlit service giving reason codes,
  covered by 41 automated tests on GitHub Actions.

After the full run, add one verified result to the second or third bullet (for example the test AUC
or the approval rate and bad rate at the cut-off), copied exactly from `reports/results/`.

Use these bullets only once the GitHub project is uploaded and you can confidently explain the
methodology in interview.
