-- Step 3: derived features, built only from information known on the application date.
-- The same calculations live in build_features() in src/credit_risk/features.py, which the API
-- uses for new applicants; tests/test_sql_matches_python.py checks the two give the same answer.

CREATE OR REPLACE TABLE model_features AS
SELECT
    *,
    (fico_range_low + fico_range_high) / 2.0                        AS fico_mid,
    date_diff('month', earliest_cr_date, issue_date)                AS credit_history_months,
    CASE WHEN annual_inc > 0 THEN loan_amnt / annual_inc END        AS loan_to_income
FROM population;
