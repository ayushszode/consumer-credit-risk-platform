"""Shared test fixtures. Everything uses synthetic loans, never real borrower data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from credit_risk.config import load_config  # noqa: E402
from credit_risk.synthetic import write_raw_file  # noqa: E402


@pytest.fixture(scope="session")
def tmp_project(tmp_path_factory):
    """A config whose data/artifact/report folders live in a temporary directory."""
    base = tmp_path_factory.mktemp("project")
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    for key, sub in [("raw_file", "raw/loans.csv.gz"), ("processed_dir", "processed"),
                     ("artifacts_dir", "artifacts"), ("reports_dir", "reports")]:
        cfg["data"][key] = str(base / sub)
    (base / "raw").mkdir()
    write_raw_file(base / "raw" / "loans.csv.gz", n=12_000, seed=1)
    path = base / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="session")
def cfg(tmp_project):
    return load_config(tmp_project)


@pytest.fixture(scope="session")
def population(cfg):
    from credit_risk.data import load_population_pandas
    pop, audit = load_population_pandas(cfg["data"]["raw_file"], cfg, chunksize=4_000)
    return pop, audit


@pytest.fixture(scope="session")
def pipeline_run(tmp_project):
    """Run the full pipeline once on synthetic data; later tests reuse the outputs."""
    from credit_risk.pipeline import run
    return run(tmp_project, synthetic=True, engine="pandas", n_synthetic=15_000)


@pytest.fixture(scope="session")
def bundle(pipeline_run, cfg):
    from credit_risk.service import load_bundle
    return load_bundle(Path(cfg["data"]["artifacts_dir"]) / "model_bundle.joblib")
