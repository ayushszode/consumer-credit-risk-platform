-- Step 1: read the raw file as text and convert the columns we need into proper types.
-- Everything is read as VARCHAR first so one odd value can't break the whole load.
-- The LendingClub file ends with a few summary lines, which have no numeric id; they are dropped.

CREATE OR REPLACE VIEW raw_loans AS
SELECT *
FROM read_csv('{raw_file}', header = true, all_varchar = true, null_padding = true, ignore_errors = true);

CREATE OR REPLACE TABLE clean_loans AS
SELECT
    TRY_CAST(id AS BIGINT)                                            AS id,
    TRY_CAST(loan_amnt AS DOUBLE)                                     AS loan_amnt,
    TRY_CAST(funded_amnt AS DOUBLE)                                   AS funded_amnt,
    TRY_CAST(regexp_extract(term, '(\d+)', 1) AS INTEGER)             AS term_months,
    TRY_CAST(trim(replace(int_rate, '%', '')) AS DOUBLE)              AS int_rate,
    TRY_CAST(installment AS DOUBLE)                                   AS installment,
    trim(grade)                                                       AS grade,
    trim(sub_grade)                                                   AS sub_grade,
    CASE
        WHEN trim(emp_length) LIKE '<%' THEN 0
        ELSE TRY_CAST(regexp_extract(emp_length, '(\d+)', 1) AS INTEGER)
    END                                                               AS emp_length_years,
    trim(home_ownership)                                              AS home_ownership,
    TRY_CAST(annual_inc AS DOUBLE)                                    AS annual_inc,
    trim(verification_status)                                         AS verification_status,
    CAST(TRY_STRPTIME(trim(issue_d), '%b-%Y') AS DATE)                AS issue_date,
    trim(loan_status)                                                 AS loan_status,
    trim(purpose)                                                     AS purpose,
    trim(addr_state)                                                  AS addr_state,
    TRY_CAST(dti AS DOUBLE)                                           AS dti,
    TRY_CAST(delinq_2yrs AS DOUBLE)                                   AS delinq_2yrs,
    CAST(TRY_STRPTIME(trim(earliest_cr_line), '%b-%Y') AS DATE)       AS earliest_cr_date,
    TRY_CAST(fico_range_low AS DOUBLE)                                AS fico_range_low,
    TRY_CAST(fico_range_high AS DOUBLE)                               AS fico_range_high,
    TRY_CAST(inq_last_6mths AS DOUBLE)                                AS inq_last_6mths,
    TRY_CAST(mths_since_last_delinq AS DOUBLE)                        AS mths_since_last_delinq,
    TRY_CAST(open_acc AS DOUBLE)                                      AS open_acc,
    TRY_CAST(pub_rec AS DOUBLE)                                       AS pub_rec,
    TRY_CAST(revol_bal AS DOUBLE)                                     AS revol_bal,
    TRY_CAST(trim(replace(revol_util, '%', '')) AS DOUBLE)            AS revol_util,
    TRY_CAST(total_acc AS DOUBLE)                                     AS total_acc,
    TRY_CAST(policy_code AS DOUBLE)                                   AS policy_code,
    trim(application_type)                                            AS application_type,
    TRY_CAST(mort_acc AS DOUBLE)                                      AS mort_acc,
    TRY_CAST(pub_rec_bankruptcies AS DOUBLE)                          AS pub_rec_bankruptcies,
    TRY_CAST(bc_util AS DOUBLE)                                       AS bc_util,
    TRY_CAST(acc_open_past_24mths AS DOUBLE)                          AS acc_open_past_24mths,
    TRY_CAST(num_actv_rev_tl AS DOUBLE)                               AS num_actv_rev_tl,
    TRY_CAST(tot_cur_bal AS DOUBLE)                                   AS tot_cur_bal,
    TRY_CAST(total_rev_hi_lim AS DOUBLE)                              AS total_rev_hi_lim,
    TRY_CAST(percent_bc_gt_75 AS DOUBLE)                              AS percent_bc_gt_75,
    TRY_CAST(mo_sin_rcnt_tl AS DOUBLE)                                AS mo_sin_rcnt_tl,
    TRY_CAST(mths_since_recent_inq AS DOUBLE)                         AS mths_since_recent_inq,
    -- outcome-period fields: only used to estimate loss given default, never as model inputs
    TRY_CAST(total_rec_prncp AS DOUBLE)                               AS total_rec_prncp,
    TRY_CAST(recoveries AS DOUBLE)                                    AS recoveries,
    TRY_CAST(collection_recovery_fee AS DOUBLE)                       AS collection_recovery_fee
FROM raw_loans
WHERE TRY_CAST(id AS BIGINT) IS NOT NULL;
