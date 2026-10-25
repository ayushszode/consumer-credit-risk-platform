-- Step 2: decide which loans belong in the modelling population, and why the others don't.
-- Each loan gets the FIRST rule it fails (NULL = eligible), so the audit adds up exactly.
-- Must stay in line with exclusion_reason() in src/credit_risk/data.py.

CREATE OR REPLACE TABLE population_flagged AS
SELECT
    *,
    CASE
        WHEN issue_date IS NULL OR loan_status IS NULL OR term_months IS NULL THEN 'missing_key_fields'
        WHEN coalesce(policy_code, 1) <> 1                                   THEN 'policy_code_not_1'
        WHEN term_months <> {term_months}                                    THEN 'not_36_month_term'
        WHEN issue_date < DATE '{issue_start}' OR issue_date > DATE '{issue_end}' THEN 'issued_outside_window'
        WHEN upper(coalesce(application_type, '{application_type}')) <> '{application_type}'
                                                                             THEN 'not_individual_application'
        WHEN loan_status NOT IN ({good_statuses}, {bad_statuses})            THEN 'unresolved_outcome'
        ELSE NULL
    END AS exclusion_reason
FROM clean_loans;

-- Target: 1 if the loan was charged off or defaulted at any point in its 36-month life, 0 if fully paid.
-- This is a lifetime default flag, NOT a 12-month probability of default.
CREATE OR REPLACE TABLE population AS
SELECT
    * EXCLUDE (exclusion_reason),
    CASE WHEN loan_status IN ({bad_statuses}) THEN 1 ELSE 0 END AS bad
FROM population_flagged
WHERE exclusion_reason IS NULL;
