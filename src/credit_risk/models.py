"""The challenger model and the lender-grade benchmark.

Challenger: scikit-learn's HistGradientBoostingClassifier.
  Why this one rather than LightGBM or XGBoost? It is a fast histogram-based gradient boosting
  model of the same family, it handles missing values and categories natively, and it ships
  with scikit-learn, so the project has one fewer dependency to install and maintain. If it ever
  became the bottleneck, swapping in LightGBM would be a small change.

Benchmark: the same model type trained on the applicant features PLUS LendingClub's own grade
and interest rate. It is not a candidate for approving loans (those fields come from the
lender's existing decision), but it shows how much of the risk the lender's own process was
already capturing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def gradient_boosting(numeric: list[str], categorical: list[str], learning_rate=0.05, max_leaf_nodes=31,
                      min_samples_leaf=200, max_iter=600, random_state=42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), categorical),
    ], verbose_feature_names_out=False)
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        max_iter=max_iter, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, categorical_features=[False] * len(numeric) + [True] * len(categorical),
        random_state=random_state,
    )
    return Pipeline([("prep", pre), ("model", model)])


def tune_gradient_boosting(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], categorical: list[str],
                           grid: list[dict], max_iter: int, random_state: int = 42) -> tuple[Pipeline, pd.DataFrame]:
    """Try each setting, pick the one with the best validation AUC. Returns (model, results)."""
    cols = numeric + categorical
    rows, best, best_auc = [], None, -1.0
    for params in grid:
        m = gradient_boosting(numeric, categorical, max_iter=max_iter, random_state=random_state, **params)
        m.fit(train[cols], train["bad"])
        auc = roc_auc_score(valid["bad"], m.predict_proba(valid[cols])[:, 1])
        rows.append({**params, "iterations_used": int(m.named_steps["model"].n_iter_), "validation_auc": auc})
        if auc > best_auc:
            best, best_auc = m, auc
    return best, pd.DataFrame(rows)


def permutation_importance_auc(model, X: pd.DataFrame, y: pd.Series, features: list[str], n_repeats: int = 3,
                               seed: int = 0) -> pd.DataFrame:
    """Drop in AUC when one feature's values are shuffled. Bigger drop = more important."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for f in features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": f, "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
