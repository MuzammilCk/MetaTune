from __future__ import annotations

from typing import Any, Dict, List, Optional


def recommend_algorithms(dna: Dict[str, Any], hyperparameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    task_type = dna.get("task_type", "classification")
    n_instances = float(dna.get("n_instances", 0) or 0)
    n_features = float(dna.get("n_features", 0) or 0)
    dimensionality = float(dna.get("dimensionality", 0) or 0)
    sparsity = float(dna.get("sparsity", 0) or 0)
    entropy = float(dna.get("target_entropy", 0) or 0)
    avg_corr = float(dna.get("avg_correlation", 0) or 0)

    candidates: List[Dict[str, Any]] = []

    def add(algorithm_id: str, label: str, reason: str, deployable: bool = True):
        candidates.append(
            {
                "id": algorithm_id,
                "label": label,
                "reason": reason,
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
            )

        if sparsity > 0.25 or entropy > 1.0:
            add(
                "gboost",
                "Gradient Boosting",
                "Dataset complexity is high (entropy/sparsity); boosting handles non-linear decision boundaries well.",
            )

        add(
            "rf",
            "Random Forest",
            "Strong default for tabular classification; robust to scaling and mixed feature interactions.",
        )

        add(
            "pytorch_mlp",
            "Neural Network (PyTorch)",
            "Use when you want adaptive training dynamics and you expect complex feature interactions.",
            deployable=True,
        )
    else:
        if avg_corr > 0.6 or n_features > 2000:
            add(
                "ridge",
                "Ridge Regression",
                "Correlated or high-dimensional features detected; Ridge is stable and easy to deploy.",
            )

        if sparsity > 0.25 or entropy > 0.0:
            add(
                "gboost_reg",
                "Gradient Boosting Regressor",
                "Non-linear patterns likely; boosting performs well on tabular regression.",
            )

        add(
            "rf_reg",
            "Random Forest Regressor",
            "Strong default for tabular regression; captures non-linearities without heavy tuning.",
        )

        add(
            "pytorch_mlp",
            "Neural Network (PyTorch)",
            "Use when you want adaptive training dynamics and you expect complex feature interactions.",
            deployable=True,
        )

    seen = set()
    ordered: List[Dict[str, Any]] = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        ordered.append(c)

    return ordered
