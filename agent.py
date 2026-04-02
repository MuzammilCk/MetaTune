import os
import json
import time
import argparse
import uuid
import hashlib
from typing import Dict, Any, List, Optional
from enum import Enum
import sys

# Local imports
try:
    from data_analyzer import DatasetAnalyzer
    from brain import MetaLearner
    from engine import DynamicTrainer
    from bilevel import BilevelOptimizer, BilevelConfig
    import algorithm_recommender
except ImportError as e:
    print(f"❌ Error: Missing component files. {e}")
    sys.exit(1)

# ==========================================
# 0. Numpy JSON Encoder
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ==========================================
# 1. Action Schemas & Constraints
# ==========================================

class ActionType(str, Enum):
    INSPECT_DATASET = "inspect_dataset"
    PROPOSE_SEARCH_SPACE = "propose_search_space"
    RUN_TRIAL = "run_trial"
    DIAGNOSE_FAILURE = "diagnose_failure"
    REVISE_STRATEGY = "revise_strategy"

class AgentState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    CRITIQUING = "critiquing"
    COMPLETED = "completed"
    FAILED = "failed"

class FailureType(str, Enum):
    DATA = "data_error"
    TRAINING = "training_error"
    APPROVAL = "approval_error"
    IO = "io_error"
    UNKNOWN = "unknown_error"

# ==========================================
# 2. Episodic Memory Manager
# ==========================================

class EpisodicMemory:
    def __init__(self, memory_file="episodic_memory.json"):
        self.memory_file = memory_file
        self.state = self._new_state()
        self.load()

    def _new_state(self) -> Dict[str, Any]:
        return {
            "run_id": str(uuid.uuid4()),
            "data_path": None,
            "dataset_fingerprint": None,
            "dataset_dna": None,
            "predicted_params": None,
            "recommended_algorithms": None,
            "actions_taken": [],
            "trial_results": None,
            "final_metric": None,
            "status": AgentState.IDLE.value,
            "failure_reason": None,
            "timestamp": time.time()
        }

    def generate_fingerprint(self, data_path: str) -> str:
        if not os.path.exists(data_path):
            return "unknown"
        # Using file size and modified time as a rough fingerprint
        stat = os.stat(data_path)
        raw_sig = f"{data_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(raw_sig.encode()).hexdigest()

    def load(self, force_new=False):
        if force_new:
            self.state = self._new_state()
            self.save()
            print(f"📂 [Memory] Starting fresh run {self.state['run_id']}")
            return

        if not force_new and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    saved_state = json.load(f)
                    if saved_state.get("status") not in [AgentState.COMPLETED.value, AgentState.FAILED.value]:
                        self.state = saved_state
                        print(f"📂 [Memory] Resuming partial run {self.state['run_id']} from {self.memory_file}")
                    else:
                        print(f"📂 [Memory] Previous run completed. Generating new memory.")
            except Exception as e:
                print(f"⚠️ [Memory] Failed to load memory: {e}. Starting fresh.")
    
    def save(self):
        try:
            self.state["timestamp"] = time.time()
            with open(self.memory_file, 'w') as f:
                json.dump(self.state, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            print(f"⚠️ [Memory] Failed to save memory: {e}")

    def log_action(self, action: ActionType, status: str, result: Any = "", failure_type: Optional[FailureType] = None):
        event = {
            "action": action.value,
            "status": status,
            "result": result if isinstance(result, str) else json.dumps(result, cls=NumpyEncoder),
            "details": result if isinstance(result, dict) else {"message": str(result)},
            "failure_type": failure_type.value if failure_type else None,
            "timestamp": time.time()
        }
        self.state["actions_taken"].append(event)
        self.save()

# ==========================================
# 3. Core Agent (Planner, Executor, Critic)
# ==========================================

class MetaTuneAgent:
    def __init__(self, data_path: str, target_col: Optional[str] = None, metric_threshold: float = 0.0, use_bilevel: bool = False, force_new: bool = False, approval_mode: str = "manual", memory_file: str = "episodic_memory.json"):
        self.data_path = data_path
        self.target_col = target_col
        self.metric_threshold = metric_threshold
        self.use_bilevel = use_bilevel
        self.approval_mode = approval_mode
        
        self.memory = EpisodicMemory(memory_file=memory_file)
        if force_new:
            self.memory.load(force_new=True)
        
        # Check tracking consistency
        fingerprint = self.memory.generate_fingerprint(data_path)
        if self.memory.state.get("dataset_fingerprint") and self.memory.state["dataset_fingerprint"] != fingerprint:
            print(f"⚠️ [Agent] Dataset has changed since last run. Starting memory fresh.")
            self.memory.load(force_new=True)

        self.memory.state["data_path"] = data_path
        self.memory.state["dataset_fingerprint"] = fingerprint
        self.memory.save()
        
        self.meta_learner = MetaLearner()

    def request_approval(self, prompt: str, auto_approve: bool = False, default_response: bool = False) -> bool:
        if self.approval_mode == "full-auto":
            return True
        if self.approval_mode == "semi-auto":
            return auto_approve or default_response
        if auto_approve:
            return True

        if not sys.stdin.isatty():
            print(f"⚠️ [APPROVAL GATE] Non-interactive shell detected. Using default response: {default_response}")
            return default_response

        print(f"\n✋ [APPROVAL GATE] {prompt}")
        while True:
            response = input("   Approve? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            if response in ['n', 'no']:
                return False

    def _classify_failure(self, exc: Exception) -> FailureType:
        msg = str(exc).lower()
        if any(k in msg for k in ["csv", "dataset", "column", "data", "load"]):
            return FailureType.DATA
        if any(k in msg for k in ["permission", "approve", "input"]):
            return FailureType.APPROVAL
        if any(k in msg for k in ["file", "path", "write", "read", "json"]):
            return FailureType.IO
        if any(k in msg for k in ["train", "optimizer", "metric", "epoch"]):
            return FailureType.TRAINING
        return FailureType.UNKNOWN

    def planner(self) -> ActionType:
        """Rule-based goal decomposition."""
        actions = [a["action"] for a in self.memory.state["actions_taken"] if a["status"] == "success"]
        
        if ActionType.INSPECT_DATASET.value not in actions:
            return ActionType.INSPECT_DATASET
        elif ActionType.PROPOSE_SEARCH_SPACE.value not in actions:
            return ActionType.PROPOSE_SEARCH_SPACE
        elif ActionType.RUN_TRIAL.value not in actions:
            return ActionType.RUN_TRIAL
        elif self.memory.state.get("status") == AgentState.FAILED.value:
            if ActionType.DIAGNOSE_FAILURE.value not in [a["action"] for a in self.memory.state["actions_taken"]]:
                return ActionType.DIAGNOSE_FAILURE
            return ActionType.REVISE_STRATEGY
        else:
            return None # Finished main execution flow

    def executor(self, action: ActionType) -> bool:
        """Executes the mapped schema routines."""
        print(f"\n🤖 [Executor] Executing action: {action.value}")
        self.memory.state["status"] = AgentState.EXECUTING.value
        self.memory.save()
        
        try:
            if action == ActionType.INSPECT_DATASET:
                analyzer = DatasetAnalyzer(self.data_path, target_col=self.target_col)
                if not analyzer.load_data():
                    raise ValueError("Failed to load dataset for analysis.")
                dna = analyzer.analyze()
                self.memory.state["dataset_dna"] = dna
                self.memory.log_action(action, "success", {
                    "message": "Dataset analysis completed",
                    "dna_feature_count": len(dna),
                })
                
            elif action == ActionType.PROPOSE_SEARCH_SPACE:
                dna = self.memory.state.get("dataset_dna")
                # Ensure MetaLearner respects KB limit internally or heuristics apply
                params = self.meta_learner.predict(dna)
                algos = algorithm_recommender.recommend_algorithms(dna)
                self.memory.state["predicted_params"] = params
                self.memory.state["recommended_algorithms"] = algos
                
                # Check for MetaBrain memory overwrite approval
                if self.meta_learner.knowledge_base_ready:
                    # Optional: train brain slightly with existing memory
                    if self.request_approval("MetaBrain is ready to learn from accumulated experience (weights update). Train MetaBrain prior to predicting?", auto_approve=False, default_response=False):
                        self.meta_learner.train()
                
                self.memory.log_action(action, "success", {
                    "message": "Search space proposed",
                    "predicted_params": params,
                    "recommended_algorithms": algos,
                })

            elif action == ActionType.RUN_TRIAL:
                params = self.memory.state.get("predicted_params")
                dna = self.memory.state.get("dataset_dna")
                
                if self.use_bilevel:
                    if self.request_approval(f"Proceed with Expensive Bilevel Optimization (N=trials, Evolutionary tuning)?", auto_approve=False, default_response=False):
                        import pandas as pd
                        from sklearn.model_selection import train_test_split
                        
                        df = pd.read_csv(self.data_path)
                        target_col = self.target_col if self.target_col else df.columns[-1]
                        X = df.drop(columns=[target_col])
                        y = df[target_col]
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        optimizer = BilevelOptimizer(meta_learner=self.meta_learner, config=BilevelConfig())
                        params = optimizer.optimize(dna, X_train, y_train, X_val, y_val, dna.get("task_type", "classification"), self.data_path)
                        self.memory.state["predicted_params"] = params
                    else:
                        print("   Skipping Bilevel Optimization. Falling back to direct training.")
                
                trainer = DynamicTrainer(self.data_path, dna, params, target_col=self.target_col)
                results = trainer.run(epochs=20)
                
                self.memory.state["trial_results"] = results
                self.memory.state["final_metric"] = results.get("final_metric", 0.0)
                self.memory.log_action(action, "success", {
                    "message": "Training trial completed",
                    "final_metric": results.get("final_metric", 0.0),
                    "metric_name": results.get("metric_name"),
                })
                
                # Store experience back into KB after trials
                if self.request_approval("Store trial experience logically back to Knowledge Base?", auto_approve=True, default_response=True):
                     self.meta_learner.store_experience(dna, params, self.memory.state["final_metric"])

            elif action == ActionType.DIAGNOSE_FAILURE:
                failure = f"Metric ({self.memory.state.get('final_metric', 0.0)}) fell below threshold ({self.metric_threshold})."
                self.memory.state["failure_reason"] = failure
                print(f"   Diagnosis: {failure}")
                self.memory.log_action(action, "success", {
                    "message": "Failure diagnosed",
                    "reason": failure,
                }, failure_type=FailureType.TRAINING)

            elif action == ActionType.REVISE_STRATEGY:
                print(f"   [Revise] Suggesting fallback logic — e.g. reducing LR or expanding search space.")
                # We perturb existing params as a naive fallback or abort.
                params = self.memory.state.get("predicted_params", {})
                if 'learning_rate' in params: params['learning_rate'] /= 2.0
                self.memory.state["predicted_params"] = params
                self.memory.log_action(action, "success", {
                    "message": "Strategy revised",
                    "revision": "Halved learning rate for next trial",
                    "updated_params": params,
                })
                # For MVP, we'll mark this complete and abort out of loops for safety
                self.memory.state["status"] = AgentState.COMPLETED.value
            
            return True

        except Exception as e:
            failure_type = self._classify_failure(e)
            self.memory.log_action(action, "failed", {
                "message": str(e),
                "exception_type": type(e).__name__,
            }, failure_type=failure_type)
            self.memory.state["failure_reason"] = str(e)
            self.memory.state["status"] = AgentState.FAILED.value
            self.memory.save()
            return False

    def critic(self) -> bool:
        """Evaluates whether the executing state yielded a successful end goal."""
        metric = self.memory.state.get("final_metric")
        if metric is None: return False
        
        self.memory.state["status"] = AgentState.CRITIQUING.value
        self.memory.save()

        if metric >= self.metric_threshold:
            print(f"✅ [Critic] Model performance acceptable: {metric:.4f} >= threshold {self.metric_threshold}")
            self.memory.state["status"] = AgentState.COMPLETED.value
            self.memory.save()
            return True
        else:
            print(f"❌ [Critic] Model performance inadequate: {metric:.4f} < threshold {self.metric_threshold}")
            self.memory.state["status"] = AgentState.FAILED.value
            self.memory.save()
            return False

    def run(self):
        print(f"\n===========================================================")
        print(f"🤖 MetaTune Agentic Orchestrator [Run ID: {self.memory.state['run_id']}]")
        print(f"===========================================================\n")
        
        while self.memory.state["status"] not in [AgentState.COMPLETED.value]:
            
            # If standard loop failed, handle diagnostics
            if self.memory.state["status"] == AgentState.FAILED.value:
                action = self.planner()
                if action:
                    self.executor(action)
                else: 
                    break # Out of options
                continue
            
            self.memory.state["status"] = AgentState.PLANNING.value
            action = self.planner()
            
            if not action:
                # All primary execution steps finished, run critic
                success = self.critic()
                if success:
                    break
            else:
                success = self.executor(action)
                if not success:
                    print(f"⚠️ [Agent] Action {action.value} failed. Halting workflow.")
                    break
        
        print("\n🎉 [Agent] Flow complete. Generating final artifacts...")
        # Export logic implementation
        if self.memory.state["status"] == AgentState.COMPLETED.value:
            if self.request_approval("Export fully deployable package (.joblib/.pth)?", default_response=False):
                print("   📦 Exporting model artifacts to local directory.")
                # We would normally invoke `train_and_package` here to serialize a production artifact.
                # Simulated for the MVP hook
        
        # Report 
        with open(f"agent_run_report_{self.memory.state['run_id'][:6]}.json", "w") as f:
            json.dump(self.memory.state, f, indent=4, cls=NumpyEncoder)
        print(f"   📄 Report written to agent_run_report_{self.memory.state['run_id'][:6]}.json")

def main():
    parser = argparse.ArgumentParser(description="MetaTune Agent orchestrator")
    parser.add_argument("data", help="Path to raw CSV dataset")
    parser.add_argument("--target", help="Target column name (optional)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum acceptable metric (Critic gate)")
    parser.add_argument("--bilevel", action="store_true", help="Launch bilevel optimizer search space")
    parser.add_argument("--new", action="store_true", help="Force ignore past episodic memory and start fresh")
    parser.add_argument("--approval-mode", choices=["manual", "semi-auto", "full-auto"], default="manual", help="Approval policy: manual prompts, semi-auto defaults, or full-auto allow")
    args = parser.parse_args()

    agent = MetaTuneAgent(
        data_path=args.data, 
        target_col=args.target, 
        metric_threshold=args.threshold,
        use_bilevel=args.bilevel,
        force_new=args.new,
        approval_mode=args.approval_mode
    )
    agent.run()

if __name__ == "__main__":
    main()
