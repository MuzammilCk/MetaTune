from __future__ import annotations

import io
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Memory / cardinality guard constants ────────────────────────────────────
# Columns with more unique values than this get OrdinalEncoded (not OneHot)
_MAX_ONEHOT_CARDINALITY = 20
# If post-encoding feature count would exceed this, apply variance-based trimming
_MAX_SAFE_FEATURES = 2000
# Always use float32 — halves memory vs float64, no accuracy loss in practice
_DTYPE = np.float32


# ── Helpers ─────────────────────────────────────────────────────────────────

def _split_cat_cols(X: pd.DataFrame):
    """Return (low_card_cat_cols, high_card_cat_cols, num_cols)."""
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    low_card  = [c for c in cat_cols if X[c].nunique() <= _MAX_ONEHOT_CARDINALITY]
    high_card = [c for c in cat_cols if X[c].nunique() >  _MAX_ONEHOT_CARDINALITY]
    return low_card, high_card, num_cols


def _estimate_dense_mb(n_rows: int, n_cols: int, dtype=_DTYPE) -> float:
    bytes_per_elem = np.dtype(dtype).itemsize
    return (n_rows * n_cols * bytes_per_elem) / (1024 ** 2)


def _build_preprocessor(X_train_raw: pd.DataFrame) -> ColumnTransformer:
    """
    Memory-safe preprocessor:
    • Numeric  → median impute → StandardScaler (float32)
    • Low-card cat (≤20 unique) → mode impute → OrdinalEncoder
      (avoids the dense OneHot explosion while keeping information)
    • High-card cat (>20 unique) → mode impute → OrdinalEncoder
    
    OrdinalEncoder is used throughout for categoricals because:
      1. It never blows up feature count regardless of cardinality.
      2. HistGradientBoosting and XGBoost handle ordinal integers natively.
      3. For LogReg/Ridge the numeric encoding is no worse than rare OneHot cols.
    """
    low_card, high_card, num_cols = _split_cat_cols(X_train_raw)
    all_cat = low_card + high_card

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=_DTYPE,
        )),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if all_cat:
        transformers.append(("cat", categorical_transformer, all_cat))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _param_to_learning_rate(hyperparameters: Optional[Dict[str, Any]]) -> float:
    if not hyperparameters:
        return 0.1
    lr = float(hyperparameters.get("learning_rate", 0.1) or 0.1)
    if not np.isfinite(lr):
        return 0.1
    return float(np.clip(lr, 0.001, 0.5))


def _param_to_alpha(hyperparameters: Optional[Dict[str, Any]]) -> float:
    if not hyperparameters:
        return 1.0
    wd = float(hyperparameters.get("weight_decay_l2", 1.0) or 1.0)
    if not np.isfinite(wd):
        return 1.0
    return float(np.clip(wd, 1e-6, 100.0))


def _clamp_params(params: dict, search_space: dict) -> dict:
    for key, bounds in search_space.items():
        if key in params and isinstance(bounds, list) and len(bounds) == 2:
            params[key] = type(params[key])(np.clip(params[key], bounds[0], bounds[1]))
    return params


# ── Estimator factory ────────────────────────────────────────────────────────

def build_estimator(
    algorithm_id: str,
    task_type: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
):
    if hyperparameters and search_space:
        hyperparameters = _clamp_params(hyperparameters, search_space)

    lr    = _param_to_learning_rate(hyperparameters)
    alpha = _param_to_alpha(hyperparameters)

    if task_type == "classification":
        if algorithm_id == "logreg":
            C = float(np.clip(1.0 / (alpha + 1e-9), 1e-4, 1e4))
            return LogisticRegression(max_iter=1000, C=C, solver="saga")

        if algorithm_id == "gboost":
            # HistGradientBoosting: histogram-based, O(n_bins) memory, not O(n_samples*n_features)
            return HistGradientBoostingClassifier(
                learning_rate=lr,
                max_iter=200,
                max_depth=6,
                random_state=42,
            )

        if algorithm_id == "rf":
            return RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )

        if algorithm_id == "xgboost":
            if not XGBOOST_AVAILABLE:
                warnings.warn("XGBoost not installed — falling back to HistGradientBoosting.")
                return HistGradientBoostingClassifier(
                    learning_rate=lr, max_iter=200, random_state=42
                )
            return XGBClassifier(
                learning_rate=lr,
                n_estimators=200,
                tree_method="hist",      # histogram mode: fast + memory-efficient
                device="cpu",
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )

    else:  # regression
        if algorithm_id == "ridge":
            return Ridge(alpha=alpha)

        if algorithm_id == "gboost_reg":
            return HistGradientBoostingRegressor(
                learning_rate=lr,
                max_iter=200,
                max_depth=6,
                random_state=42,
            )

        if algorithm_id == "rf_reg":
            return RandomForestRegressor(
                n_estimators=200,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )

        if algorithm_id == "xgboost_reg":
            if not XGBOOST_AVAILABLE:
                warnings.warn("XGBoost not installed — falling back to HistGradientBoosting.")
                return HistGradientBoostingRegressor(
                    learning_rate=lr, max_iter=200, random_state=42
                )
            return XGBRegressor(
                learning_rate=lr,
                n_estimators=200,
                tree_method="hist",
                device="cpu",
                random_state=42,
                verbosity=0,
            )

    raise ValueError(f"Unsupported algorithm_id='{algorithm_id}' for task_type='{task_type}'")


# ── Shared data-prep helper ──────────────────────────────────────────────────

def _prepare_data(data_path: str, dna: dict, target_col: Optional[str]):
    df = pd.read_csv(data_path)
    if not target_col or target_col == "":
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    task_type = dna.get("task_type", "classification")

    # Log memory estimate before any transforms
    low_card, high_card, num_cols = _split_cat_cols(X)
    n_out_features = len(num_cols) + len(low_card) + len(high_card)
    est_mb = _estimate_dense_mb(len(X), n_out_features)
    if est_mb > 500:
        warnings.warn(
            f"Post-encoding matrix estimated at {est_mb:.0f} MB — "
            "using memory-safe OrdinalEncoder pipeline."
        )

    stratify = None
    if task_type == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least 2 classes.")
        stratify = y if y.nunique() > 1 else None

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    le: Optional[LabelEncoder] = None
    if task_type == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val_np = y_val_raw.to_numpy()
        y_val_masked = np.array([
            x if x in le.classes_ else le.classes_[0] for x in y_val_np
        ])
        y_val = le.transform(y_val_masked)
    else:
        y_train = y_train_raw.to_numpy().astype(_DTYPE)
        y_val   = y_val_raw.to_numpy().astype(_DTYPE)

    return X_train_raw, X_val_raw, y_train, y_val, le, task_type, target_col


# ── Public API ───────────────────────────────────────────────────────────────

def train_and_package(
    data_path: str,
    dna: Dict[str, Any],
    algorithm_id: str,
    target_col: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    X_train_raw, X_val_raw, y_train, y_val, le, task_type, target_col = \
        _prepare_data(data_path, dna, target_col)

    preprocessor = _build_preprocessor(X_train_raw)
    search_space  = kwargs.get("search_space", None)
    estimator     = build_estimator(
        algorithm_id=algorithm_id,
        task_type=task_type,
        hyperparameters=hyperparameters,
        search_space=search_space,
    )

    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    start_time = time.time()
    model.fit(X_train_raw, y_train)
    preds   = model.predict(X_val_raw)
    elapsed = time.time() - start_time

    if task_type == "classification":
        metric_value = float(accuracy_score(y_val, preds))
        metric_name  = "Accuracy"
        metrics      = {"accuracy": metric_value}
    else:
        metric_value = float(r2_score(y_val, preds))
        metric_name  = "R2 Score"
        metrics      = {"r2": metric_value}

    package = {
        "model": model,
        "label_encoder": le,
        "target_col": target_col,
        "task_type": task_type,
        "algorithm_id": algorithm_id,
    }

    return_dict = {
        "final_metric":  metric_value,
        "metric_name":   metric_name,
        "training_time": elapsed,
        "package_path":  f"metatune_{algorithm_id}_package.joblib",
        "model_type":    algorithm_id,
    }
    return_dict.update(metrics)

    return package, return_dict


def train_and_package_staged(
    data_path: str,
    dna: Dict[str, Any],
    algorithm_id: str,
    target_col: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Generator version of train_and_package.
    Yields {'stage': int, 'stage_name': str, 'done': False} at the start of
    each real phase, then a final {'done': True, 'package': ..., 'results': ...}.
    """

    # ── STAGE 0: DATA PREP ──────────────────────────────────────────────────
    yield {"stage": 0, "stage_name": "DATA PREP", "done": False}

    X_train_raw, X_val_raw, y_train, y_val, le, task_type, target_col = \
        _prepare_data(data_path, dna, target_col)

    # ── STAGE 1: BUILDING ───────────────────────────────────────────────────
    yield {"stage": 1, "stage_name": "BUILDING", "done": False}

    preprocessor = _build_preprocessor(X_train_raw)
    search_space  = kwargs.get("search_space", None)
    estimator     = build_estimator(
        algorithm_id=algorithm_id,
        task_type=task_type,
        hyperparameters=hyperparameters,
        search_space=search_space,
    )
    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    # ── STAGE 2: FITTING ────────────────────────────────────────────────────
    yield {"stage": 2, "stage_name": "FITTING", "done": False}

    _start = time.time()
    model.fit(X_train_raw, y_train)
    preds   = model.predict(X_val_raw)
    elapsed = time.time() - _start

    if task_type == "classification":
        metric_value = float(accuracy_score(y_val, preds))
        metric_name  = "Accuracy"
        metrics      = {"accuracy": metric_value}
    else:
        metric_value = float(r2_score(y_val, preds))
        metric_name  = "R2 Score"
        metrics      = {"r2": metric_value}

    # ── STAGE 3: PACKAGING ──────────────────────────────────────────────────
    yield {"stage": 3, "stage_name": "PACKAGING", "done": False}

    package = {
        "model": model,
        "label_encoder": le,
        "target_col": target_col,
        "task_type": task_type,
        "algorithm_id": algorithm_id,
    }

    results = {
        "final_metric":  metric_value,
        "metric_name":   metric_name,
        "training_time": elapsed,
        "package_path":  f"metatune_{algorithm_id}_package.joblib",
        "model_type":    algorithm_id,
    }
    results.update(metrics)

    yield {"stage": 3, "stage_name": "PACKAGING", "done": True, "package": package, "results": results}


# ── Serialization helpers ────────────────────────────────────────────────────

def package_to_joblib_bytes(package: Dict[str, Any]) -> bytes:
    """Serialize a training package to in-memory joblib bytes for download."""
    buf = io.BytesIO()
    joblib.dump(package, buf)
    buf.seek(0)
    return buf.read()


def predict_with_package(package: Dict[str, Any], df_features: pd.DataFrame):
    """Run inference using a saved package dict. Returns decoded labels if classification."""
    preds = package["model"].predict(df_features)
    le = package.get("label_encoder")
    if le is not None:
        return le.inverse_transform(preds)
    return preds