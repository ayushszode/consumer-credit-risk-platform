"""Two extra challenger models: XGBoost and a small PyTorch neural network.

They answer the question a validator or interviewer always asks: "did you try XGBoost, or a
neural network, and was it worth it?" Each one is trained on the 2012-13 loans, tuned or
early-stopped on 2014, and scored on the same 2015 loans as the scorecard, so the comparison is fair.

They are reported, not deployed. The champion rule in pipeline.py still chooses between the
scorecard and the scikit-learn gradient boosting model, and the API and app only use those two.
If an extra challenger clearly wins on the real data, that is a finding for the validation
review, not a silent swap.

Both libraries are optional. If one isn't installed the model is skipped and the run says so:
    pip install -r requirements-extras.txt --extra-index-url https://download.pytorch.org/whl/cpu
"""
from __future__ import annotations

import importlib.util
import logging
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

log = logging.getLogger(__name__)

EXTRA_MODELS = ("xgboost", "torch_mlp")
LIBRARY = {"xgboost": "xgboost", "torch_mlp": "torch"}


def is_available(library: str) -> bool:
    """True if the library can be imported (checked without importing it)."""
    return importlib.util.find_spec(library) is not None


def _onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)


# ---------- XGBoost ----------

class XGBoostChallenger:
    """XGBoost with early stopping on the validation year.

    Numeric features go in as they are (XGBoost learns which way to send missing values);
    the three categorical features are one-hot encoded.
    """

    name = "xgboost"

    def __init__(self, numeric: list[str], categorical: list[str], learning_rate: float = 0.05,
                 max_depth: int = 4, min_child_weight: float = 50, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_lambda: float = 1.0, n_estimators: int = 2000,
                 early_stopping_rounds: int = 50, random_state: int = 42, n_jobs: int = -1):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.params = dict(learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                           subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda)
        self.n_estimators, self.early_stopping_rounds = n_estimators, early_stopping_rounds
        self.random_state, self.n_jobs = random_state, n_jobs

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return self.prep_.transform(df[self.numeric + self.categorical]).astype(np.float32)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "XGBoostChallenger":
        import xgboost as xgb

        self.prep_ = ColumnTransformer([("num", "passthrough", self.numeric), ("cat", _onehot(), self.categorical)],
                                       verbose_feature_names_out=False).fit(train[self.numeric + self.categorical])
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state, n_jobs=self.n_jobs, **self.params)
        self.model_.fit(self._x(train), train["bad"].to_numpy(),
                        eval_set=[(self._x(valid), valid["bad"].to_numpy())], verbose=False)
        return self

    @property
    def iterations_used(self) -> int:
        return int(self.model_.best_iteration) + 1

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._x(df))


def tune_xgboost(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], categorical: list[str],
                 grid: list[dict], n_estimators: int = 2000, early_stopping_rounds: int = 50,
                 random_state: int = 42) -> tuple[XGBoostChallenger, pd.DataFrame]:
    """Try each setting, keep the one with the best validation AUC (same rule as the main challenger)."""
    rows, best, best_auc = [], None, -1.0
    for params in grid:
        m = XGBoostChallenger(numeric, categorical, n_estimators=n_estimators,
                              early_stopping_rounds=early_stopping_rounds, random_state=random_state,
                              **params).fit(train, valid)
        auc = roc_auc_score(valid["bad"], m.predict_proba(valid)[:, 1])
        rows.append({**params, "iterations_used": m.iterations_used, "validation_auc": auc})
        if auc > best_auc:
            best, best_auc = m, auc
    return best, pd.DataFrame(rows)


# ---------- PyTorch neural network ----------

def mlp_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Neural networks need inputs on similar scales and no missing values.

    numeric  -> median fill, then a quantile transform to a normal shape (tames income and
                balance outliers without hand-picked caps)
    missing  -> a 0/1 flag for every numeric column that had gaps in the training years
    category -> one-hot
    """
    numeric_steps = Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                            subsample=200_000, random_state=0))])
    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("missing", MissingIndicator(features="missing-only", error_on_new=False), numeric),
        ("cat", _onehot(), categorical),
    ], verbose_feature_names_out=False)


def _build_network(n_inputs: int, hidden: list[int], dropout: float):
    import torch.nn as nn

    layers, width = [], n_inputs
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, 1))  # one output: the log-odds of default
    return nn.Sequential(*layers)


class TorchMLPChallenger:
    """A small feed-forward network (two hidden layers by default) trained with binary cross-entropy.

    Early stopping: after every pass over the training loans, check log loss on the validation
    year; keep the best weights and stop once it hasn't improved for `patience` passes.
    No class weighting, so the outputs stay usable as probabilities.
    """

    name = "torch_mlp"

    def __init__(self, numeric: list[str], categorical: list[str], hidden: tuple[int, ...] = (64, 32),
                 dropout: float = 0.1, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 1024, max_epochs: int = 30, patience: int = 3, random_state: int = 42):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.hidden, self.dropout = list(hidden), dropout
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.random_state = random_state

    def _tensor(self, df: pd.DataFrame):
        import torch
        return torch.tensor(self.prep_.transform(df[self.numeric + self.categorical]), dtype=torch.float32)

    def _predict(self, X) -> np.ndarray:
        import torch
        self.net_.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net_(X)).squeeze(1).numpy().astype(float)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "TorchMLPChallenger":
        import torch

        torch.manual_seed(self.random_state)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        rng = np.random.default_rng(self.random_state)

        self.prep_ = mlp_preprocessor(self.numeric, self.categorical).fit(train[self.numeric + self.categorical])
        X_train, X_valid = self._tensor(train), self._tensor(valid)
        y_train = torch.tensor(train["bad"].to_numpy(), dtype=torch.float32)
        y_valid = valid["bad"].to_numpy()

        self.net_ = _build_network(X_train.shape[1], self.hidden, self.dropout)
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_loss, best_state, waited, history = np.inf, None, 0, []
        n = len(X_train)
        for epoch in range(1, self.max_epochs + 1):
            self.net_.train()
            order = torch.from_numpy(rng.permutation(n))
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self.net_(X_train[idx]).squeeze(1), y_train[idx])
                loss.backward()
                optimiser.step()
            p_valid = np.clip(self._predict(X_valid), 1e-6, 1 - 1e-6)
            valid_loss = float(log_loss(y_valid, p_valid, labels=[0, 1]))
            history.append({"epoch": epoch, "validation_log_loss": valid_loss,
                            "validation_auc": float(roc_auc_score(y_valid, p_valid))})
            if valid_loss < best_loss - 1e-5:
                best_loss, waited = valid_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.history_ = pd.DataFrame(history)
        self.best_epoch_ = int(self.history_.loc[self.history_["validation_log_loss"].idxmin(), "epoch"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self._predict(self._tensor(df))
        return np.column_stack([1 - p, p])


# ---------- run both ----------

def fit_extra_challengers(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], categorical: list[str],
                          settings: dict | None, random_state: int = 42
                          ) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Train each enabled extra challenger whose library is installed.

    Returns (models by name, one status row per model, extra tables such as training history).
    """
    settings = settings or {}
    models, rows, tables = {}, [], {}
    for name in EXTRA_MODELS:
        s = dict(settings.get(name) or {})
        if not s.pop("enabled", False):
            rows.append({"model": name, "status": "disabled in config"})
            continue
        if not is_available(LIBRARY[name]):
            log.info("%s skipped: %s is not installed", name, LIBRARY[name])
            rows.append({"model": name, "status": f"skipped: {LIBRARY[name]} not installed"})
            continue
        log.info("Training extra challenger %s", name)
        if name == "xgboost":
            grid = s.pop("grid", [{}])
            model, grid_table = tune_xgboost(train, valid, numeric, categorical, grid, random_state=random_state, **s)
            tables["xgboost_grid"] = grid_table
            detail = {**model.params, "iterations_used": model.iterations_used}
        else:
            model = TorchMLPChallenger(numeric, categorical, random_state=random_state, **s).fit(train, valid)
            tables["torch_mlp_training"] = model.history_
            detail = {"hidden": "x".join(map(str, model.hidden)), "best_epoch": model.best_epoch_,
                      "epochs_run": len(model.history_)}
        auc = float(roc_auc_score(valid["bad"], model.predict_proba(valid)[:, 1]))
        models[name] = model
        rows.append({"model": name, "status": "trained", "validation_auc": auc,
                     "settings": "; ".join(f"{k}={v}" for k, v in detail.items())})
    return models, pd.DataFrame(rows), tables
