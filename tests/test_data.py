import pandas as pd

from credit_risk.data import EXCLUSION_ORDER, clean_raw, split_summary, time_split
from credit_risk.synthetic import make_raw_loans


def test_footer_rows_are_dropped():
    raw = make_raw_loans(n=200, add_footer=True)
    clean = clean_raw(raw)
    assert len(clean) == 200
    assert clean["id"].notna().all()


def test_text_fields_are_parsed():
    raw = make_raw_loans(n=300)
    clean = clean_raw(raw)
    assert set(clean["term_months"].dropna().unique()) <= {36, 60}
    assert clean["issue_date"].notna().all()
    assert clean["emp_length_years"].between(0, 10).all()
    lt1 = raw["emp_length"].iloc[:300] == "< 1 year"
    assert (clean.loc[lt1.to_numpy(), "emp_length_years"] == 0).all()


def test_audit_adds_up(population):
    _, audit = population
    counts = dict(zip(audit["step"], audit["loans"]))
    assert counts["total_rows"] == sum(counts[s] for s in EXCLUSION_ORDER) + counts["eligible"]


def test_population_rules(population, cfg):
    pop, _ = population
    p = cfg["population"]
    assert (pop["term_months"] == 36).all()
    assert pop["issue_date"].between(pd.Timestamp(p["issue_start"]), pd.Timestamp(p["issue_end"])).all()
    assert pop["loan_status"].isin(p["good_statuses"] + p["bad_statuses"]).all()
    assert (pop["application_type"].str.upper() == "INDIVIDUAL").all()


def test_target_mapping(population, cfg):
    pop, _ = population
    bad_statuses = set(cfg["population"]["bad_statuses"])
    assert (pop["bad"] == pop["loan_status"].isin(bad_statuses).astype(int)).all()


def test_splits_are_in_time_order_and_do_not_overlap(population, cfg):
    pop, _ = population
    parts = time_split(pop, cfg)
    assert parts["train"]["issue_date"].max() < parts["validation"]["issue_date"].min()
    assert parts["validation"]["issue_date"].max() < parts["test"]["issue_date"].min()
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert split_summary(parts)["loans"].sum() == len(pop)


def test_raw_file_found_when_name_differs(tmp_path):
    from credit_risk.data import find_raw_file
    sub = tmp_path / "accepted_2007_to_2018q4.csv"
    sub.mkdir()
    real = sub / "accepted_2007_to_2018Q4.csv"
    real.write_text("id\n1\n")
    assert find_raw_file(tmp_path / "accepted_2007_to_2018Q4.csv.gz") == real
