from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False



def _build_preprocessor(X_train_raw: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns
    cat_cols = X_train_raw.select_dtypes(exclude=[np.number]).columns

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )


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

def build_estimator(algorithm_id: str, task_type: str, hyperparameters: Optional[Dict[str, Any]] = None, search_space: Optional[Dict[str, Any]] = None):
    if hyperparameters and search_space:
        hyperparameters = _clamp_params(hyperparameters, search_space)
        
    lr = _param_to_learning_rate(hyperparameters)
    alpha = _param_to_alpha(hyperparameters)

    if task_type == "classification":
        if algorithm_id == "logreg":
            C = float(np.clip(1.0 / (alpha + 1e-9), 1e-4, 1e4))
            return LogisticRegression(max_iter=500, C=C)
        if algorithm_id == "gboost":
            return GradientBoostingClassifier(learning_rate=lr, random_state=42)
        if algorithm_id == "rf":
            return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        if algorithm_id == "xgboost":
            if not XGBOOST_AVAILABLE:
                print("⚠️  XGBoost not installed. Falling back to GradientBoosting.")
                return GradientBoostingClassifier(learning_rate=lr, random_state=42)
            return XGBClassifier(learning_rate=lr, random_state=42, 
                                 eval_metric='logloss', verbosity=0)
    else:
        if algorithm_id == "ridge":
            return Ridge(alpha=alpha)
        if algorithm_id == "gboost_reg":
            return GradientBoostingRegressor(learning_rate=lr, random_state=42)
        if algorithm_id == "rf_reg":
            return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        if algorithm_id == "xgboost_reg":
            if not XGBOOST_AVAILABLE:
                print("⚠️  XGBoost not installed. Falling back to GradientBoosting.")
                return GradientBoostingRegressor(learning_rate=lr, random_state=42)
            return XGBRegressor(learning_rate=lr, random_state=42, verbosity=0)

    raise ValueError(f"Unsupported algorithm_id='{algorithm_id}' for task_type='{task_type}'")


def train_and_package(
    data_path: str,
    dna: Dict[str, Any],
    algorithm_id: str,
    target_col: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    df = pd.read_csv(data_path)
    if target_col is None or target_col == "":
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    task_type = dna.get("task_type", "classification")

    le: Optional[LabelEncoder] = None
    if task_type == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val_np = y_val_raw.to_numpy()
        y_val_masked = np.array([x if x in le.classes_ else le.classes_[0] for x in y_val_np])
        y_val = le.transform(y_val_masked)
    else:
        y_train = y_train_raw.to_numpy().astype(np.float32)
        y_val = y_val_raw.to_numpy().astype(np.float32)

    preprocessor = _build_preprocessor(X_train_raw)
    
    search_space = kwargs.get('search_space', None)
    estimator = build_estimator(algorithm_id=algorithm_id, task_type=task_type, hyperparameters=hyperparameters, search_space=search_space)

    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    
    import time
    start_time = time.time()
    
    # Handle the extreme case where the ENTIRE dataset natively only has 1 class
    if task_type == 'classification':
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # If even the raw data only has 1 class, we must artificially inject a second
            # to prevent Sklearn from crashing outright
            dummy_class = 1 if unique_classes[0] == 0 else 0
            
            # Inject one dummy opposite class into the training split
            X_train_raw.iloc[0] = X_train_raw.iloc[0] # Keep same features
            y_train[0] = dummy_class
            
            # Re-fit the LabelEncoder to ensure it knows about the dummy class
            if le is not None and dummy_class not in le.transform(le.classes_):
                new_classes = np.append(le.classes_, [f"dummy_{dummy_class}"])
                le.classes_ = np.unique(new_classes)
            
    model.fit(X_train_raw, y_train)

    preds = model.predict(X_val_raw)
    elapsed = time.time() - start_time

    if task_type == "classification":
        metric_value = float(accuracy_score(y_val, preds))
        metric_name = "Accuracy"
        metrics = {"accuracy": metric_value}
    else:
        metric_value = float(r2_score(y_val, preds))
        metric_name = "R2 Score"
        metrics = {"r2": metric_value}

    package = {
        "model": model,
        "label_encoder": le,
        "target_col": target_col,
        "task_type": task_type,
        "algorithm_id": algorithm_id,
    }
    
    return_dict = {
        "final_metric": metric_value,
        "metric_name": metric_name,
        "training_time": elapsed,
        "package_path": f"metatune_{algorithm_id}_package.joblib",
        "model_type": algorithm_id
    }
    
    return_dict.update(metrics)

    return package, return_dict


def package_to_joblib_bytes(package: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    joblib.dump(package, buf)
    buf.seek(0)
    return buf.read()


def predict_with_package(package: Dict[str, Any], df_features: pd.DataFrame):
    preds = package["model"].predict(df_features)
    le = package.get("label_encoder")
    if le is not None:
        return le.inverse_transform(preds)
    return preds
