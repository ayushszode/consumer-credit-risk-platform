"""Reading the raw LendingClub file and building the modelling population.

Two ways to run the same steps:
  * DuckDB + the SQL files in sql/ (preferred: reads the 2m+ row file without loading it all)
  * pandas, reading in chunks (fallback when DuckDB isn't installed, and used in the tests)

Both apply the same rules, in the same order, and `tests/test_sql_matches_python.py` checks
they agree.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .features import RAW_COLUMNS, RAW_NUMERIC, build_features

log = logging.getLogger(__name__)

EXCLUSION_ORDER = [
    "missing_key_fields", "policy_code_not_1", "not_36_month_term", "issued_outside_window",
    "not_individual_application", "unresolved_outcome",
]


# ---------- cleaning ----------

def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the text columns of the raw file into proper types (pandas path)."""
    df = pd.DataFrame(index=raw.index)
    df["id"] = pd.to_numeric(raw["id"], errors="coerce")
    for col in RAW_NUMERIC:
        if col in raw:
            df[col] = pd.to_numeric(raw[col].astype("string").str.replace("%", "", regex=False).str.strip(),
                                    errors="coerce").astype("float64")
        else:
            df[col] = np.nan
    df["term_months"] = pd.to_numeric(raw["term"].astype("string").str.extract(r"(\d+)")[0],
                                      errors="coerce").astype("float64")
    df["issue_date"] = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_date"] = pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y", errors="coerce")
    emp = raw["emp_length"].astype("string").str.strip()
    years = pd.to_numeric(emp.str.extract(r"(\d+)")[0], errors="coerce").astype("float64")
    years = years.where(~emp.str.startswith("<", na=False).to_numpy(dtype=bool), 0.0)
    df["emp_length_years"] = years
    for col in ["grade", "sub_grade", "home_ownership", "verification_status", "loan_status", "purpose",
                "addr_state", "application_type"]:
        df[col] = raw[col].astype("string").str.strip() if col in raw else pd.NA
    return df[df["id"].notna()].reset_index(drop=True)


def exclusion_reason(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """First population rule each loan fails (missing value = eligible). Mirrors sql/02_population.sql."""
    p = cfg["population"]
    resolved = set(p["good_statuses"]) | set(p["bad_statuses"])
    start, end = pd.Timestamp(p["issue_start"]), pd.Timestamp(p["issue_end"])
    rules = [
        ("missing_key_fields", df["issue_date"].isna() | df["loan_status"].isna() | df["term_months"].isna()),
        ("policy_code_not_1", df["policy_code"].fillna(1) != 1),
        ("not_36_month_term", df["term_months"] != p["term_months"]),
        ("issued_outside_window", (df["issue_date"] < start) | (df["issue_date"] > end)),
        ("not_individual_application", df["application_type"].fillna(p["application_type"]).str.upper()
         != p["application_type"].upper()),
        ("unresolved_outcome", ~df["loan_status"].isin(resolved)),
    ]
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, mask in rules:
        reason = reason.mask(reason.isna() & mask.fillna(False).astype(bool), name)
    return reason


def add_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bad = set(cfg["population"]["bad_statuses"])
    out = df.copy()
    out["bad"] = out["loan_status"].isin(bad).astype(int)
    return out


# ---------- loading ----------

def _raw_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0, compression="infer").columns.tolist()


def load_population_pandas(path: str | Path, cfg: dict, chunksize: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw file in chunks, clean it and keep the eligible population.

    Returns (population, audit) where audit counts how many loans each rule removed.
    """
    path = Path(path)
    header = _raw_header(path)
    usecols = [c for c in RAW_COLUMNS if c in header]
    missing = sorted(set(RAW_COLUMNS) - set(header))
    if missing:
        log.warning("Columns not in the file (filled with missing values): %s", missing)
    kept, counts = [], pd.Series(0, index=["total_rows"] + EXCLUSION_ORDER + ["eligible"], dtype="int64")
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=chunksize, compression="infer",
                             low_memory=False):
        for col in set(RAW_COLUMNS) - set(chunk.columns):
            chunk[col] = pd.NA
        clean = clean_raw(chunk)
        counts["total_rows"] += len(clean)
        reason = exclusion_reason(clean, cfg)
        for name, n in reason.value_counts().items():
            counts[name] += int(n)
        eligible = clean[reason.isna()]
        counts["eligible"] += len(eligible)
        kept.append(eligible)
    pop = pd.concat(kept, ignore_index=True)
    pop = build_features(add_target(pop, cfg))
    audit = counts.rename_axis("step").reset_index(name="loans")
    return pop, audit


def load_population_duckdb(path: str | Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same result as the pandas path, using the SQL files and DuckDB."""
    import duckdb  # imported here so the rest of the project works without it

    p = cfg["population"]
    params = {
        "raw_file": str(Path(path)).replace("'", "''"),
        "term_months": p["term_months"],
        "issue_start": p["issue_start"],
        "issue_end": p["issue_end"],
        "good_statuses": ", ".join(f"'{s}'" for s in p["good_statuses"]),
        "bad_statuses": ", ".join(f"'{s}'" for s in p["bad_statuses"]),
        "application_type": p["application_type"].upper(),
    }
    con = duckdb.connect()
    for name in ["01_clean.sql", "02_population.sql", "03_features.sql"]:
        sql = (PROJECT_ROOT / "sql" / name).read_text().format(**params)
        con.execute(sql)
    audit_df = con.execute("""
        SELECT coalesce(exclusion_reason, 'eligible') AS step, count(*) AS loans
        FROM population_flagged GROUP BY 1
    """).df()
    total = int(con.execute("SELECT count(*) FROM clean_loans").fetchone()[0])
    pop = con.execute("SELECT * FROM model_features ORDER BY id").df()
    con.close()
    counts = dict(zip(audit_df["step"], audit_df["loans"]))
    audit = pd.DataFrame({"step": ["total_rows"] + EXCLUSION_ORDER + ["eligible"],
                          "loans": [total] + [int(counts.get(s, 0)) for s in EXCLUSION_ORDER]
                          + [int(counts.get("eligible", 0))]})
    pop["issue_date"] = pd.to_datetime(pop["issue_date"])
    pop["earliest_cr_date"] = pd.to_datetime(pop["earliest_cr_date"])
    return build_features(pop), audit


def find_raw_file(configured: str | Path) -> Path:
    """The configured file, or else any accepted-loans CSV (zipped or not) in the same folder.

    Kaggle downloads sometimes arrive unzipped, or inside a folder named after the file, so we
    also look one level down.
    """
    path = Path(configured)
    if path.exists():
        return path
    folder = path.parent
    candidates = [*folder.glob("*"), *folder.glob("*/*")] if folder.exists() else []
    matches = sorted((p for p in candidates if p.is_file() and p.name.lower().startswith("accepted")
                      and p.name.lower().endswith((".csv", ".csv.gz"))),
                     key=lambda p: (not p.name.lower().endswith(".gz"), str(p)))
    if matches:
        log.info("Using %s (configured file %s not found)", matches[0], path.name)
        return matches[0]
    raise FileNotFoundError(f"Raw data not found at {path}. See data/README.md for how to download it.")


def load_population(cfg: dict, engine: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = find_raw_file(cfg["data"]["raw_file"])
    if engine in ("auto", "duckdb"):
        try:
            pop, audit = load_population_duckdb(path, cfg)
            return pop, audit, "duckdb"
        except ImportError:
            if engine == "duckdb":
                raise
            log.info("DuckDB not installed, using the pandas reader instead")
    pop, audit = load_population_pandas(path, cfg)
    return pop, audit, "pandas"


# ---------- sampling and splits ----------

def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Keep the same share of loans from every issue month and outcome."""
    keys = [df["issue_date"].dt.to_period("M"), df["bad"]]
    return (df.groupby(keys, group_keys=False, observed=True)
              .sample(frac=frac, random_state=seed)
              .sort_values("issue_date").reset_index(drop=True))


def time_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Train on the oldest loans, validate on the next year, test on the most recent."""
    s = cfg["splits"]
    train_end, valid_end = pd.Timestamp(s["train_end"]), pd.Timestamp(s["validation_end"])
    parts = {
        "train": df[df["issue_date"] <= train_end],
        "validation": df[(df["issue_date"] > train_end) & (df["issue_date"] <= valid_end)],
        "test": df[df["issue_date"] > valid_end],
    }
    if df["id"].duplicated().any():
        raise ValueError("Duplicate loan ids found; splits would not be clean")
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def split_summary(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([{
        "split": name, "loans": len(d), "bad_loans": int(d["bad"].sum()),
        "bad_rate": d["bad"].mean() if len(d) else np.nan,
        "first_issue": d["issue_date"].min().strftime("%Y-%m") if len(d) else "",
        "last_issue": d["issue_date"].max().strftime("%Y-%m") if len(d) else "",
    } for name, d in parts.items()])
