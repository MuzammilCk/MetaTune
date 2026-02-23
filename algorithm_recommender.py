from __future__ import annotations

from typing import Any, Dict, List, Optional


def recommend_algorithms(dna: Dict[str, Any] = None, hyperparameters: Optional[Dict[str, Any]] = None) -> dict:
    if dna is None:
        dna = {}
        
    task_type = dna.get("task_type", "classification")
    n_instances = float(dna.get("n_instances", 0) or 0)
    n_features = float(dna.get("n_features", 0) or 0)
    dimensionality = float(dna.get("dimensionality", 0) or 0)
    sparsity = float(dna.get("sparsity", 0) or 0)
    entropy = float(dna.get("target_entropy", 0) or 0)
    avg_corr = float(dna.get("avg_correlation", 0) or 0)

    candidates: List[Dict[str, Any]] = []

    def add(algorithm_id: str, label: str, reason: str, search_space: dict, deployable: bool = True):
        candidates.append(
            {
                "id": algorithm_id,
                "label": label,
                "reason": reason,
                "search_space": search_space,
                "deployable": deployable,
                "task_type": task_type,
            }
        )

    if task_type == "classification":
        if n_features > 2000 or dimensionality > 0.15:
            add(
                "logreg",
                "Logistic Regression",
                "High dimensionality detected; linear models often generalize well and deploy easily.",
                {"C": [0.001, 100.0], "max_iter": [100, 1000]}
            )

        if sparsity > 0.25 or entropy > 1.0:
            add(
                "gboost",
                "Gradient Boosting",
                "Dataset complexity is high (entropy/sparsity); boosting handles non-linear decision boundaries well.",
                {"n_estimators": [50, 300], "max_depth": [2, 8], "learning_rate": [0.01, 0.3]}
            )

        if n_instances > 1000 and n_features < 200 and sparsity < 0.5:
            add(
                "xgboost",
                "XGBoost",
                "Strong tabular algorithm; good for medium-to-large datasets with mixed feature types.",
                {"n_estimators": [50, 500], "max_depth": [3, 10], "learning_rate": [0.01, 0.3], "subsample": [0.6, 1.0]}
            )

        add(
            "rf",
            "Random Forest",
            "Strong default for tabular classification; robust to scaling and mixed feature interactions.",
            {"n_estimators": [50, 500], "max_depth": [3, 20], "min_samples_split": [2, 20]}
        )

        add(
            "pytorch_mlp",
            "Neural Network (PyTorch)",
            "Use when you want adaptive training dynamics and you expect complex feature interactions.",
            {"lr": [1e-4, 1e-1], "dropout": [0.0, 0.5], "weight_decay_l2": [1e-5, 1e-1]},
            deployable=True,
        )
    else:
        if avg_corr > 0.6 or n_features > 2000:
            add(
                "ridge",
                "Ridge Regression",
                "Correlated or high-dimensional features detected; Ridge is stable and easy to deploy.",
                {"alpha": [0.0001, 100.0]}
            )

        if sparsity > 0.25 or entropy > 0.0:
            add(
                "gboost_reg",
                "Gradient Boosting Regressor",
                "Non-linear patterns likely; boosting performs well on tabular regression.",
                {"n_estimators": [50, 300], "max_depth": [2, 8], "learning_rate": [0.01, 0.3]}
            )
            
        if n_instances > 1000 and n_features < 200 and sparsity < 0.5:
            add(
                "xgboost_reg",
                "XGBoost",
                "Strong tabular algorithm; good for medium-to-large datasets with mixed feature types.",
                {"n_estimators": [50, 500], "max_depth": [3, 10], "learning_rate": [0.01, 0.3], "subsample": [0.6, 1.0]}
            )

        add(
            "rf_reg",
            "Random Forest Regressor",
            "Strong default for tabular regression; captures non-linearities without heavy tuning.",
            {"n_estimators": [50, 500], "max_depth": [3, 20], "min_samples_split": [2, 20]}
        )

        add(
            "pytorch_mlp",
            "Neural Network (PyTorch)",
            "Use when you want adaptive training dynamics and you expect complex feature interactions.",
            {"lr": [1e-4, 1e-1], "dropout": [0.0, 0.5], "weight_decay_l2": [1e-5, 1e-1]},
            deployable=True,
        )

    seen = set()
    ordered: List[Dict[str, Any]] = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        ordered.append(c)
        
    base = 20
    if float(dna.get("n_samples", 0.0) or 0) > 10000.0 or n_instances > 10000.0:
        base = 50
    if float(dna.get("task_difficulty_score", 0.0) or 0) > 1.5:
        base += 20
    
    n_trials_budget = min(base, 100)

    if not dna: # Backward compatibility
        n_trials_budget = 30

    return {
        "recommendations": ordered,
        "n_trials_budget": int(n_trials_budget)
    }
