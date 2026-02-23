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

from brain import MetaLearner
from engine import DynamicTrainer

@dataclass
class BilevelConfig:
    min_trials: int = 5           # Vizier: tuning_min_num_trials
    max_outer_iterations: int = 10 # Vizier: (max-min)/num_per_tuning
    population_size: int = 3      # Vizier: pool_size
    perturbation: float = 0.1     # Vizier: FireflyAlgorithmConfig.perturbation
    perturbation_lower_bound: float = 0.01  # Vizier: perturbation_lower_bound

class BilevelOptimizer:
    def __init__(self, meta_learner, config: BilevelConfig = None):
        self.meta_learner = meta_learner
        self.config = config or BilevelConfig()
        self.trial_history = []  # List of {"hyperparams": dict, "val_metric": float}
        self.state = "INITIALIZE"
    
    def optimize(self, dataset_dna: dict, X_train, y_train, X_val, y_val, task_type: str) -> dict:
        """Run bilevel optimization. Returns best hyperparams found."""
        self.current_dna = dataset_dna
        
        print(f"🔄 Starting Bilevel Optimization (Vizier Inspired) for {self.config.max_outer_iterations} iterations.")
        
        # STATE 1 - INITIALIZE
        self.state = "INITIALIZE"
        print(f"   [STATE: {self.state}] Gathering initial {self.config.min_trials} trials...")
        for _ in range(self.config.min_trials):
            hyperparams = self.meta_learner.predict(dataset_dna)
            val_metric = self._evaluate_hyperparams(hyperparams, X_train, y_train, X_val, y_val, task_type)
            self.trial_history.append({"hyperparams": hyperparams, "val_metric": val_metric})
            self.meta_learner.store_experience(dataset_dna, hyperparams, val_metric)

        # STATE 2 - TUNE
        self.state = "TUNE"
        print(f"   [STATE: {self.state}] Starting evolutionary search for {self.config.max_outer_iterations} outer iterations...")
        search_hint_str = dataset_dna.get("vizier_search_space_hint", "{}")
        try:
            search_hint = ast.literal_eval(search_hint_str) if search_hint_str else {}
        except Exception:
            search_hint = {}

        for iteration in range(self.config.max_outer_iterations):
            best_anchor = self.get_best_hyperparams()
            print(f"      Outer Iteration {iteration+1}/{self.config.max_outer_iterations} - Perturbing from best anchor")
            
            candidates = []
            for _ in range(self.config.population_size):
                candidate_hp = self._perturb_hyperparams(best_anchor, search_hint)
                val_metric = self._evaluate_hyperparams(candidate_hp, X_train, y_train, X_val, y_val, task_type)
                candidates.append({"hyperparams": candidate_hp, "val_metric": val_metric})
                self.meta_learner.store_experience(dataset_dna, candidate_hp, val_metric)
            
            self.trial_history.extend(candidates)

        # STATE 3 - USE_BEST
        self.state = "USE_BEST"
        print(f"   [STATE: {self.state}] Training meta-learner on accumulated experience...")
        self.meta_learner.train()
        
        best_found = self.get_best_hyperparams()
        print(f"✅ Bilevel Optimization Complete. Best metric found.")
        return best_found
    
    def _evaluate_hyperparams(self, hyperparams: dict, X_train, y_train, X_val, y_val, task_type: str) -> float:
        """Inner loop: train with given hyperparams, return val metric."""
        # DynamicTrainer internally reads temp.csv. We use it to match MetaTune architecture.
        trainer = DynamicTrainer(data_path="temp.csv", dataset_dna=self.current_dna, hyperparameters=hyperparams)
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
