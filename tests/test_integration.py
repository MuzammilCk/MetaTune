"""
Integration Test: Algorithm Recommender <-> Sklearn Engine Contract
===================================================================
WHAT THIS TESTS:
    Every algorithm ID that algorithm_recommender.py can output MUST have
    a corresponding case in sklearn_engine.build_estimator(). If these two
    files ever go out of sync, this test fails immediately at dev time
    instead of crashing at runtime in front of a user.

WHY THIS EXISTS:
    A previous edit added XGBoost to the recommender but forgot to add it
    to the engine. The app crashed with:
        ValueError: Unsupported algorithm_id='xgboost_reg' for task_type='regression'
    This test makes that class of bug impossible to ship undetected.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithm_recommender import recommend_algorithms
from sklearn_engine import build_estimator


# DNA profiles that trigger the widest possible set of recommendations
# so no algorithm ID can hide from this test.
CLASSIFICATION_DNA = {
    "task_type": "classification",
    "n_instances": 5000,
    "n_features": 20,
    "sparsity": 0.3,           # triggers gboost
    "target_entropy": 1.2,     # triggers gboost
    "avg_correlation": 0.3,
    "dimensionality": 0.004,
}

REGRESSION_DNA = {
    "task_type": "regression",
    "n_instances": 5000,
    "n_features": 20,
    "sparsity": 0.3,           # triggers gboost_reg
    "target_entropy": 0.0,
    "avg_correlation": 0.7,    # triggers ridge
    "dimensionality": 0.004,
}

# IDs that are intentionally NOT handled by sklearn_engine
# because they are routed to engine_stream.py instead
PYTORCH_IDS = {"pytorch_mlp"}


class TestAlgorithmEngineContract(unittest.TestCase):
    """
    Contract test suite. If any test here fails, it means:
    algorithm_recommender.py and sklearn_engine.py are out of sync.
    Fix: add the missing algorithm_id case to build_estimator() in sklearn_engine.py.
    """

    def _assert_all_supported(self, recommendations: list, task_type: str):
        """Helper: verify every recommended ID can be built by the engine."""
        for rec in recommendations:
            algo_id = rec["id"]

            if algo_id in PYTORCH_IDS:
                continue  # PyTorch path is handled separately, skip

            with self.subTest(algorithm_id=algo_id, task_type=task_type):
                try:
                    build_estimator(
                        algorithm_id=algo_id,
                        task_type=task_type,
                        hyperparameters=None
                    )
                except ValueError:
                    self.fail(
                        f"\n\n❌ CONTRACT BROKEN\n"
                        f"   algorithm_recommender.py outputs: '{algo_id}' "
                        f"for task_type='{task_type}'\n"
                        f"   sklearn_engine.build_estimator() has NO case for it.\n\n"
                        f"FIX: Open sklearn_engine.py → build_estimator() → "
                        f"add a case for algorithm_id='{algo_id}' "
                        f"under the '{task_type}' branch."
                    )

    def test_classification_algorithms_all_supported(self):
        """Every algorithm recommended for classification must be buildable."""
        result = recommend_algorithms(CLASSIFICATION_DNA)
        recommendations = result["recommendations"]
        self.assertGreater(len(recommendations), 0,
            "Recommender returned no algorithms for classification DNA.")
        self._assert_all_supported(recommendations, "classification")

    def test_regression_algorithms_all_supported(self):
        """Every algorithm recommended for regression must be buildable."""
        result = recommend_algorithms(REGRESSION_DNA)
        recommendations = result["recommendations"]
        self.assertGreater(len(recommendations), 0,
            "Recommender returned no algorithms for regression DNA.")
        self._assert_all_supported(recommendations, "regression")

    def test_return_format_is_dict_with_recommendations_key(self):
        """Recommender must return a dict with 'recommendations' key, not a plain list."""
        result = recommend_algorithms(CLASSIFICATION_DNA)
        self.assertIsInstance(result, dict,
            "recommend_algorithms() must return a dict, not a list.")
        self.assertIn("recommendations", result,
            "Return dict must have a 'recommendations' key.")
        self.assertIn("n_trials_budget", result,
            "Return dict must have a 'n_trials_budget' key.")

    def test_every_recommendation_has_required_keys(self):
        """Each recommendation dict must have id, label, reason, search_space."""
        result = recommend_algorithms(CLASSIFICATION_DNA)
        for rec in result["recommendations"]:
            for required_key in ["id", "label", "reason", "search_space"]:
                self.assertIn(required_key, rec,
                    f"Recommendation missing required key: '{required_key}'. "
                    f"Got: {list(rec.keys())}")

    def test_xgboost_classification_specifically(self):
        """
        Regression test for the specific bug that caused the crash.
        XGBoost classification was added to recommender but not engine.
        This test ensures it never regresses.
        """
        try:
            estimator = build_estimator("xgboost", "classification", hyperparameters=None)
            self.assertIsNotNone(estimator)
        except ValueError:
            self.fail(
                "❌ REGRESSION: 'xgboost' for classification is not supported in "
                "sklearn_engine.build_estimator(). This was the original bug."
            )

    def test_xgboost_regression_specifically(self):
        """
        Regression test for the specific bug that caused the crash.
        XGBoost regression ('xgboost_reg') crashed the app on car dataset.
        This test ensures it never regresses.
        """
        try:
            estimator = build_estimator("xgboost_reg", "regression", hyperparameters=None)
            self.assertIsNotNone(estimator)
        except ValueError:
            self.fail(
                "❌ REGRESSION: 'xgboost_reg' for regression is not supported in "
                "sklearn_engine.build_estimator(). This was the original crash bug."
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
