"""
Bilevel Optimization Engine for MetaTune (Vizier-Inspired)
==========================================================

Outer Loop (meta-learner / Vizier-style):
  -> brain.predict(dataset_dna)          -> TrialSuggestion equivalent
  -> engine.run(hyperparams)             -> Trial evaluation
  -> store result as "completed trial"   -> Measurement equivalent  
  -> brain.train() with new experience   -> Designer.update() equivalent
  -> repeat until convergence or budget  -> MetaLearningConfig.tuning_max_num_trials

Usage in pipeline.py:
# optimizer = BilevelOptimizer(meta_learner=self.meta_learner)
# best_params = optimizer.optimize(dataset_dna, X_train, y_train, X_val, y_val, task_type)
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
import copy
import ast

from metatune_logging import get_logger

logger = get_logger(__name__)

from brain import MetaLearner
from engine import DynamicTrainer

@dataclass
class BilevelConfig:
    min_trials: int = 5           # Vizier: tuning_min_num_trials
    max_outer_iterations: int = 5  # Reduced from 10 for faster CI/demo runs
    population_size: int = 2       # Reduced from 3 for faster CI/demo runs
    perturbation: float = 0.1     # Vizier: FireflyAlgorithmConfig.perturbation
    perturbation_lower_bound: float = 0.01  # Vizier: perturbation_lower_bound

class BilevelOptimizer:
    def __init__(self, meta_learner, config: BilevelConfig = None):
        self.meta_learner = meta_learner
        self.config = config or BilevelConfig()
        self.trial_history = []  # List of {"hyperparams": dict, "val_metric": float}
        self.state = "INITIALIZE"
    
    def optimize(self, dataset_dna: dict, X_train, y_train, X_val, y_val, task_type: str, data_path: str = None, target_col: str = None, df=None) -> dict:
        """Run bilevel optimization. Returns best hyperparams found.

        NOTE: X_train/y_train/X_val/y_val are accepted for backward
        compatibility with existing callers but are NOT used for training —
        see _evaluate_hyperparams. Each inner trial gets its own fresh
        DynamicTrainer (own preprocessing fit + split) against data_path/df.
        target_col and df (a pre-loaded, ideally already-cleaned DataFrame)
        are what actually reach the trainer; pass them explicitly rather
        than relying on DynamicTrainer's own "last column" fallback, which
        will pick the wrong column for most real datasets.
        """
        self.current_dna = dataset_dna
        if data_path is None and df is None:
            raise ValueError("BilevelOptimizer.optimize requires data_path and/or df.")
        self._data_path = data_path
        self._target_col = target_col
        self._df = df
        
        logger.info(f"🔄 Starting Bilevel Optimization (Vizier Inspired) for {self.config.max_outer_iterations} iterations.")
        
        # STATE 1 - INITIALIZE
        self.state = "INITIALIZE"
        logger.info(f"   [STATE: {self.state}] Gathering initial {self.config.min_trials} trials...")
        for _ in range(self.config.min_trials):
            hyperparams = self.meta_learner.predict(dataset_dna)
            val_metric = self._evaluate_hyperparams(hyperparams, X_train, y_train, X_val, y_val, task_type)
            self.trial_history.append({"hyperparams": hyperparams, "val_metric": val_metric})
            self.meta_learner.store_experience(dataset_dna, hyperparams, val_metric)

        # STATE 2 - TUNE
        self.state = "TUNE"
        logger.info(f"   [STATE: {self.state}] Starting evolutionary search for {self.config.max_outer_iterations} outer iterations...")
        search_hint_str = dataset_dna.get("vizier_search_space_hint", "{}")
        if isinstance(search_hint_str, dict):
            search_hint = search_hint_str
        else:
            try:
                search_hint = ast.literal_eval(search_hint_str) if search_hint_str else {}
            except Exception:
                search_hint = {}

        for iteration in range(self.config.max_outer_iterations):
            best_anchor = self.get_best_hyperparams()
            logger.info(f"      Outer Iteration {iteration+1}/{self.config.max_outer_iterations} - Perturbing from best anchor")
            
            candidates = []
            for _ in range(self.config.population_size):
                candidate_hp = self._perturb_hyperparams(best_anchor, search_hint)
                val_metric = self._evaluate_hyperparams(candidate_hp, X_train, y_train, X_val, y_val, task_type)
                candidates.append({"hyperparams": candidate_hp, "val_metric": val_metric})
                self.meta_learner.store_experience(dataset_dna, candidate_hp, val_metric)
            
            self.trial_history.extend(candidates)

        # STATE 3 - USE_BEST
        self.state = "USE_BEST"
        logger.info(f"   [STATE: {self.state}] Training meta-learner on accumulated experience...")
        self.meta_learner.train()
        
        best_found = self.get_best_hyperparams()
        logger.info(f"✅ Bilevel Optimization Complete. Best metric found.")
        return best_found
    
    def _evaluate_hyperparams(self, hyperparams: dict, X_train, y_train, X_val, y_val, task_type: str) -> float:
        """Inner loop: train with given hyperparams, return val metric.
        NOTE: X_train/y_train/X_val/y_val are accepted for API consistency
        but not used — DynamicTrainer re-derives its own split from
        data_path/df with a fixed random_state, so an externally-supplied
        split would just be discarded anyway."""
        trainer = DynamicTrainer(
            data_path=self._data_path,
            dataset_dna=self.current_dna,
            hyperparameters=hyperparams,
            target_col=self._target_col,
            df=self._df,
        )
        result = trainer.run(epochs=10)
        return result.get("final_metric", 0.0)
    
    def _perturb_hyperparams(self, base_hyperparams: dict, search_hint: dict) -> dict:
        """Generate a perturbed candidate. Only perturb continuous params."""
        candidate = copy.deepcopy(base_hyperparams)
        continuous_keys = ['learning_rate', 'dropout', 'weight_decay_l2']
        
        for key in continuous_keys:
            if key in candidate:
                val = candidate[key]
                noise = np.random.normal(0, self.config.perturbation)
                perturbed_val = abs(val + noise)
                
                # Check for perturbation lower bound
                if perturbed_val < self.config.perturbation_lower_bound and key != 'dropout':
                    perturbed_val = float(self.config.perturbation_lower_bound)
                    
                # Clamp to search_space_hint bounds
                if key in search_hint and isinstance(search_hint[key], list) and len(search_hint[key]) == 2:
                    bounds = search_hint[key]
                    perturbed_val = np.clip(perturbed_val, bounds[0], bounds[1])
                elif key == 'dropout':
                    perturbed_val = np.clip(perturbed_val, 0.0, 0.5)
                    
                candidate[key] = float(perturbed_val)
                
        return candidate
    
    def get_best_hyperparams(self) -> dict:
        """Return hyperparams with best val_metric from trial_history."""
        if not self.trial_history:
            return {}
        # Assuming higher metric is better (Accuracy, R2)
        best_trial = max(self.trial_history, key=lambda x: x["val_metric"])
        return best_trial["hyperparams"]
