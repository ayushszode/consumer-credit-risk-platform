"""Run the whole project end to end.

    python -m credit_risk.pipeline                 # real LendingClub file (see data/README.md)
    python -m credit_risk.pipeline --sample 0.2    # 20% stratified sample, for quick development
    python -m credit_risk.pipeline --synthetic     # fake data, for demos and checks (no download)

Outputs:
    artifacts/model_bundle.joblib   - models + policy + versions, used by the API and app
    reports/results/*.csv, *.json   - every number quoted in the reports
    reports/figures/*.png           - charts
    reports/results.md              - auto-generated summary of this run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import evaluate, experiment, figures, monitoring, policy, pricing, quality
from .config import load_config
from .data import load_population, split_summary, stratified_sample, time_split
from .features import (CATEGORICAL_FEATURES, LENDER_FIELDS, MODEL_FEATURES, NUMERIC_FEATURES, assert_no_leakage,
                       feature_register)
from .models import permutation_importance_auc, tune_gradient_boosting
from .scorecard import Scorecard, iv_strength

log = logging.getLogger("credit_risk")
TARGET_DEFINITION = ("1 = loan charged off or defaulted at any point during its 36-month term; "
                     "0 = fully paid. Lifetime default flag, not a 12-month PD.")


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def choose_champion(metrics: dict, paired: dict, cal: dict) -> tuple[str, str]:
    """Documented rule: keep the scorecard unless boosting is clearly better AND still well calibrated.

    'Clearly better' = test AUC higher by at least 0.01 with the bootstrap interval above zero.
    'Well calibrated' = calibration slope between 0.8 and 1.2 on the test year.
    """
    gain, lo = paired["auc_difference"], paired["ci_low"]
    slope = cal["gbm"]["calibration_slope"]
    if gain >= 0.01 and lo > 0 and 0.8 <= slope <= 1.2:
        return "gbm", (f"Gradient boosting improved test AUC by {gain:.3f} (95% CI {lo:.3f} to {paired['ci_high']:.3f}) "
                       f"and stayed calibrated (slope {slope:.2f}), which outweighs the loss of simplicity.")
    return "scorecard", (f"The scorecard was kept: boosting changed test AUC by {gain:+.3f} "
                         f"(95% CI {lo:.3f} to {paired['ci_high']:.3f}; calibration slope {slope:.2f}). "
                         "That gain is not large or certain enough to give up a model whose every point can be explained.")


def run(config_path=None, synthetic=False, sample_frac=None, engine="auto", n_synthetic=80_000) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    seed = cfg["data"]["random_state"]
    reports = Path(cfg["data"]["reports_dir"])
    res_dir, fig_dir, art_dir = reports / "results", reports / "figures", Path(cfg["data"]["artifacts_dir"])
    for d in (res_dir, fig_dir, art_dir):
        d.mkdir(parents=True, exist_ok=True)

    note = None
    if synthetic:
        from .synthetic import write_raw_file
        path = Path(cfg["data"]["processed_dir"]) / "synthetic_loans.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            write_raw_file(path, n=n_synthetic, seed=seed)
        cfg["data"]["raw_file"] = str(path)
        note = "SYNTHETIC DEMO DATA - numbers say nothing about real credit risk"
    run_label = "synthetic demo data" if synthetic else "LendingClub accepted loans 2007-2018Q4"

    # 1. Data
    log.info("Loading population from %s", cfg["data"]["raw_file"])
    pop, audit, engine_used = load_population(cfg, engine)
    if sample_frac or cfg["data"]["sample_frac"]:
        frac = sample_frac or cfg["data"]["sample_frac"]
        pop = stratified_sample(pop, frac, seed)
        run_label += f" ({frac:.0%} stratified sample)"
    assert_no_leakage(MODEL_FEATURES)
    checks = quality.run_checks(pop)
    parts = time_split(pop, cfg)
    splits = split_summary(parts)
    train, valid, test = parts["train"], parts["validation"], parts["test"]
    log.info("Population %d loans (%s engine). Train %d / validation %d / test %d",
             len(pop), engine_used, len(train), len(valid), len(test))

    # 2. Scorecard (champion candidate)
    s = cfg["scorecard"]
    sc = Scorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, n_bins=s["n_bins"], min_bin_share=s["min_bin_share"],
                   min_iv=s["min_iv"], max_corr=s["max_corr"], C=s["C"], base_score=s["base_score"],
                   base_odds=s["base_odds"], pdo=s["pdo"]).fit(train, train["bad"])
    iv = sc.iv_table()
    iv["strength"] = iv["iv"].map(iv_strength)

    # 3. Gradient boosting challenger, and a benchmark that also sees the lender's grade
    b = cfg["boosting"]
    gbm, grid = tune_gradient_boosting(train, valid, NUMERIC_FEATURES, CATEGORICAL_FEATURES, b["grid"], b["max_iter"], seed)
    bench, _ = tune_gradient_boosting(train, valid, NUMERIC_FEATURES + ["int_rate"], CATEGORICAL_FEATURES + ["sub_grade"],
                                      b["grid"][:1], b["max_iter"], seed)

    def scores(df):
        return {"scorecard": sc.predict_proba(df)[:, 1],
                "gbm": gbm.predict_proba(df[MODEL_FEATURES])[:, 1],
                "benchmark": bench.predict_proba(df[NUMERIC_FEATURES + ["int_rate"] + CATEGORICAL_FEATURES + ["sub_grade"]])[:, 1]}

    # 4. Evaluation on the same loans for every model
    metric_rows, cal_tables, cal_stats = [], {}, {}
    for split_name, df in [("validation", valid), ("test", test)]:
        sc_ = scores(df)
        for m, p in sc_.items():
            row = {"split": split_name, "model": m, **evaluate.summary(df["bad"], p)}
            row["auc_ci_low"], row["auc_ci_high"] = evaluate.bootstrap_ci(df["bad"], p, seed=seed)
            row.update(evaluate.calibration_slope_intercept(df["bad"], p))
            metric_rows.append(row)
            if split_name == "test":
                cal_tables[m] = evaluate.calibration_table(df["bad"], p)
                cal_stats[m] = evaluate.calibration_slope_intercept(df["bad"], p)
        if split_name == "test":
            test_scores = sc_
    metrics = pd.DataFrame(metric_rows)
    paired = evaluate.paired_auc_difference(test["bad"], test_scores["scorecard"], test_scores["gbm"], seed=seed)
    champion, reason = choose_champion(metrics, paired, cal_stats)
    test = test.assign(pd_scorecard=test_scores["scorecard"], pd_gbm=test_scores["gbm"],
                       pd_champion=test_scores[champion], score_points=sc.score(test))
    log.info("Champion: %s", champion)

    # 5. Policy and loss analysis on the test year
    pc = cfg["policy"]
    bands = policy.band_table(test, "pd_champion", pc["bands"])
    curve = policy.threshold_curve(test, "pd_champion")
    outcomes = policy.outcome_summary(test, "pd_champion", pc)
    confusion = pd.DataFrame([evaluate.confusion_at(test["bad"], test["pd_champion"], t)
                              for t in (pc["approve_max_pd"], pc["decline_min_pd"])])
    lgd_ead = pricing.estimate_lgd_ead(train)
    if cfg["pricing"]["lgd_override"] is not None:
        lgd_ead["lgd"] = cfg["pricing"]["lgd_override"]
    approved = test[test["pd_champion"] <= pc["approve_max_pd"]]
    el_sens = pricing.el_sensitivity(approved["pd_champion"], approved["loan_amnt"])
    el_point = float(pricing.expected_loss(approved["pd_champion"], approved["loan_amnt"], lgd_ead["lgd"],
                                           lgd_ead["ead_share"]).sum() / approved["loan_amnt"].sum())
    pr = cfg["pricing"]
    test["illustrative_rate"] = pricing.illustrative_rate(test["pd_champion"], lgd_ead["lgd"], lgd_ead["ead_share"],
                                                          funding_cost=pr["funding_cost"], operating_cost=pr["operating_cost"],
                                                          target_margin=pr["target_margin"])
    rate_compare = (test.groupby(policy.risk_band(test["pd_champion"], pc["bands"]))
                        .agg(loans=("bad", "size"), illustrative_rate=("illustrative_rate", "mean"),
                             lendingclub_rate=("int_rate", lambda r: r.mean() / 100))
                        .reset_index().rename(columns={"index": "band"}))

    # Offline champion vs challenger at the same selection rate
    share = (test["pd_champion"] <= pc["approve_max_pd"]).mean()
    other = "gbm" if champion == "scorecard" else "scorecard"
    other_cut = float(np.quantile(test[f"pd_{other}"], share))
    offline = experiment.offline_policy_comparison(test, "pd_champion", pc["approve_max_pd"], f"pd_{other}", other_cut)
    base_bad = float(test.loc[test["pd_champion"] <= pc["approve_max_pd"], "bad"].mean())
    sample_size = experiment.required_sample_size(base_bad, 0.01)

    # 6. Monitoring replay: validation + test years against the training years
    mc = cfg["monitoring"]
    train_scored = train.assign(pd_champion=scores(train)[champion])
    later = pd.concat([valid.assign(pd_champion=scores(valid)[champion]), test], ignore_index=True)
    replay = monitoring.quarterly_replay(train_scored, later, "pd_champion", NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                                         mc["psi_investigate"], mc["psi_action"])
    drift_test = monitoring.drift_report(train, test, NUMERIC_FEATURES, CATEGORICAL_FEATURES, mc["psi_investigate"], mc["psi_action"])
    vintages = monitoring.vintage_table(pop)
    segments = {seg: evaluate.metrics_by_segment(test, "pd_champion", seg) for seg in ["home_ownership", "purpose", "addr_state"]}
    imp_sample = valid.sample(min(len(valid), 20_000), random_state=seed)
    importance = permutation_importance_auc(gbm, imp_sample[MODEL_FEATURES], imp_sample["bad"],
                                            MODEL_FEATURES, n_repeats=2, seed=seed)

    # 7. Save artefacts
    typical = {c: (float(train[c].median()) if c in NUMERIC_FEATURES else str(train[c].mode().iloc[0])) for c in MODEL_FEATURES}
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    bundle = {"model_version": f"{champion}-{version}", "champion": champion, "champion_reason": reason,
              "scorecard": sc, "gbm": gbm, "policy": pc, "typical_values": typical, "target_definition": TARGET_DEFINITION,
              "lgd": lgd_ead["lgd"], "ead_share": lgd_ead["ead_share"], "trained_on": run_label,
              "train_window": f"{train['issue_date'].min():%Y-%m} to {train['issue_date'].max():%Y-%m}",
              "synthetic": synthetic}
    joblib.dump(bundle, art_dir / "model_bundle.joblib", compress=3)

    tables = {"population_audit": audit, "data_quality": checks, "splits": splits, "information_value": iv,
              "points_table": sc.points_table, "boosting_grid": grid, "metrics": metrics, "bands_test": bands,
              "threshold_curve_test": curve, "policy_outcomes_test": outcomes, "confusion_test": confusion,
              "el_sensitivity": el_sens, "rate_comparison": rate_compare, "offline_policy_comparison": offline,
              "monitoring_replay": replay, "drift_test_vs_train": drift_test, "vintages": vintages,
              "permutation_importance_gbm": importance, "feature_register": feature_register(),
              "missing_by_year": quality.missing_by_year(pop).reset_index(names="feature"),
              **{f"segments_{k}": v for k, v in segments.items()},
              **{f"calibration_test_{k}": v for k, v in cal_tables.items()}}
    for name, t in tables.items():
        t.to_csv(res_dir / f"{name}.csv", index=False)
    summary = {"run_label": run_label, "synthetic": synthetic, "engine": engine_used, "model_version": bundle["model_version"],
               "champion": champion, "champion_reason": reason, "paired_auc_test": paired, "lgd_ead": lgd_ead,
               "expected_loss_pct_approved_test": el_point, "approve_share_test": float(share),
               "live_experiment_loans_per_arm_for_1pt_change": sample_size, "target_definition": TARGET_DEFINITION,
               "runtime_seconds": round(time.time() - t0, 1)}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # 8. Figures
    figures.roc_chart(test["bad"], test_scores, fig_dir, note)
    figures.calibration_chart({k: cal_tables[k] for k in ("scorecard", "gbm")}, fig_dir, note)
    figures.tradeoff_chart(curve, fig_dir, note)
    figures.vintage_chart(vintages, fig_dir, note)
    figures.psi_chart(replay, mc["psi_investigate"], mc["psi_action"], fig_dir, note)
    figures.iv_chart(iv, fig_dir, note)

    # 9. Auto-generated results page
    m_test = metrics[metrics["split"] == "test"][["model", "loans", "bad_rate", "auc", "auc_ci_low", "auc_ci_high", "gini",
                                                  "ks", "brier", "log_loss", "calibration_slope"]]
    lines = [f"# Results of the latest run\n",
             f"_Generated automatically by `credit_risk.pipeline` on {datetime.now():%d %b %Y %H:%M}. Data: {run_label}._\n"]
    if synthetic:
        lines.append("> **These results come from synthetic demo data. They only show the pipeline works; "
                     "they say nothing about real credit risk.**\n")
    lines += ["## Population", _md_table(audit, "{:,.0f}"), "", "## Splits", _md_table(splits), "",
              f"**Target:** {TARGET_DEFINITION}\n",
              "## Model comparison (test year, same loans for every model)", _md_table(m_test), "",
              f"**Champion: {champion}.** {reason}\n",
              "## Risk bands (test year)", _md_table(bands), "",
              "## Policy outcomes at the configured cut-offs (test year, historical funded loans only)", _md_table(outcomes), "",
              f"Estimated LGD {lgd_ead['lgd']:.2f} and EAD {lgd_ead['ead_share']:.2f} of the original amount "
              f"(from {lgd_ead['charged_off_loans_used']:,} charged-off training loans). "
              f"Expected lifetime loss on auto-approved test loans: {el_point:.2%} of the amount lent.\n",
              "## Monitoring replay", _md_table(replay[["quarter", "loans", "score_psi", "score_status", "most_shifted_feature",
                                                         "its_psi", "mean_predicted_pd", "observed_bad_rate", "auc"]]), "",
              "## Top features by Information Value", _md_table(iv.head(10)[["feature", "iv", "strength", "status"]], "{:.4g}"), ""]
    (reports / "results.md").write_text("\n".join(lines))
    log.info("Done in %.0f s", time.time() - t0)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Consumer credit risk platform pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use generated fake data instead of the real file")
    ap.add_argument("--sample", type=float, default=None, help="stratified sample fraction, e.g. 0.2")
    ap.add_argument("--engine", choices=["auto", "duckdb", "pandas"], default="auto")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    summary = run(args.config, args.synthetic, args.sample, args.engine)
    print(json.dumps({k: summary[k] for k in ("run_label", "champion", "model_version", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
