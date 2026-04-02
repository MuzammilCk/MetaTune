import os
import json
import time
import argparse
import uuid
import hashlib
from typing import Dict, Any, List, Optional
from enum import Enum
import sys
from dataclasses import dataclass

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

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class FailureType(str, Enum):
    DATA = "data_error"
    TRAINING = "training_error"
    APPROVAL = "approval_error"
    IO = "io_error"
    UNKNOWN = "unknown_error"

@dataclass
class TaskNode:
    action: ActionType
    depends_on: List[ActionType]
    deadline_epoch: int
    retries_left: int = 1
    status: TaskStatus = TaskStatus.PENDING

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
    def __init__(self, data_path: str, target_col: Optional[str] = None, metric_threshold: float = 0.0, use_bilevel: bool = False, force_new: bool = False, approval_mode: str = "manual", memory_file: str = "episodic_memory.json", action_budget: int = 12):
        self.data_path = data_path
        self.target_col = target_col
        self.metric_threshold = metric_threshold
        self.use_bilevel = use_bilevel
        self.approval_mode = approval_mode
        self.action_budget = action_budget
        
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
        self.memory.state.setdefault("policy_trace", [])
        self.memory.save()
        
        self.meta_learner = MetaLearner()
        self.task_graph = self._init_task_graph()

    def _init_task_graph(self) -> Dict[ActionType, TaskNode]:
        return {
            ActionType.INSPECT_DATASET: TaskNode(ActionType.INSPECT_DATASET, [], deadline_epoch=2, retries_left=2),
            ActionType.PROPOSE_SEARCH_SPACE: TaskNode(ActionType.PROPOSE_SEARCH_SPACE, [ActionType.INSPECT_DATASET], deadline_epoch=5, retries_left=2),
            ActionType.RUN_TRIAL: TaskNode(ActionType.RUN_TRIAL, [ActionType.PROPOSE_SEARCH_SPACE], deadline_epoch=10, retries_left=2),
            ActionType.DIAGNOSE_FAILURE: TaskNode(ActionType.DIAGNOSE_FAILURE, [ActionType.RUN_TRIAL], deadline_epoch=11, retries_left=1),
            ActionType.REVISE_STRATEGY: TaskNode(ActionType.REVISE_STRATEGY, [ActionType.DIAGNOSE_FAILURE], deadline_epoch=12, retries_left=1),
        }

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

    def planner(self) -> Optional[Dict[str, Any]]:
        """Policy planner with simple task-graph, uncertainty, and budget awareness."""
        completed = {ActionType(a["action"]) for a in self.memory.state["actions_taken"] if a["status"] == "success"}
        actions_used = len(self.memory.state["actions_taken"])
        remaining_budget = max(0, self.action_budget - actions_used)
        uncertainty = 1.0 if self.memory.state.get("dataset_dna") is None else 0.3

        if remaining_budget <= 0:
            return None

        if self.memory.state.get("status") == AgentState.FAILED.value:
            for failed_action in [ActionType.DIAGNOSE_FAILURE, ActionType.REVISE_STRATEGY]:
                node = self.task_graph[failed_action]
                deps_ready = all(dep in completed for dep in node.depends_on)
                if deps_ready and node.retries_left >= 0 and node.status != TaskStatus.COMPLETED:
                    rationale = f"Recovery path selected: {failed_action.value} (remaining_budget={remaining_budget})."
                    self.memory.state["policy_trace"].append({"action": failed_action.value, "rationale": rationale, "timestamp": time.time()})
                    self.memory.save()
                    return {"action": failed_action, "rationale": rationale, "remaining_budget": remaining_budget, "uncertainty": uncertainty}
            return None

        candidates = []
        for action, node in self.task_graph.items():
            if action in [ActionType.DIAGNOSE_FAILURE, ActionType.REVISE_STRATEGY]:
                continue
            if node.status == TaskStatus.COMPLETED:
                continue
            deps_ready = all(dep in completed for dep in node.depends_on)
            if not deps_ready:
                continue
            urgency = max(0, node.deadline_epoch - actions_used)
            score = (2.0 if action not in completed else 0.5) + (1.0 / (urgency + 1)) + uncertainty
            candidates.append((score, action))

        if not candidates:
            return None

        _, selected = max(candidates, key=lambda x: x[0])
        rationale = f"Selected {selected.value}: deps_satisfied, uncertainty={uncertainty:.2f}, remaining_budget={remaining_budget}."
        self.memory.state["policy_trace"].append({"action": selected.value, "rationale": rationale, "timestamp": time.time()})
        self.memory.save()
        return {"action": selected, "rationale": rationale, "remaining_budget": remaining_budget, "uncertainty": uncertainty}

    def executor(self, action: ActionType, rationale: str = "") -> Dict[str, Any]:
        """Executes the mapped schema routines."""
        print(f"\n🤖 [Executor] Executing action: {action.value}")
        self.memory.state["status"] = AgentState.EXECUTING.value
        self.memory.save()
        self.task_graph[action].status = TaskStatus.IN_PROGRESS
        
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
            
            self.task_graph[action].status = TaskStatus.COMPLETED
            return {
                "success": True,
                "confidence": 0.85,
                "cost": {"actions_used": len(self.memory.state["actions_taken"])},
                "artifacts": {"action": action.value},
                "error_class": None,
                "rationale": rationale,
            }

        except Exception as e:
            failure_type = self._classify_failure(e)
            self.task_graph[action].status = TaskStatus.FAILED
            self.task_graph[action].retries_left -= 1
            self.memory.log_action(action, "failed", {
                "message": str(e),
                "exception_type": type(e).__name__,
            }, failure_type=failure_type)
            self.memory.state["failure_reason"] = str(e)
            self.memory.state["status"] = AgentState.FAILED.value
            self.memory.save()
            return {
                "success": False,
                "confidence": 0.1,
                "cost": {"actions_used": len(self.memory.state["actions_taken"])},
                "artifacts": {"action": action.value},
                "error_class": failure_type.value,
                "rationale": rationale,
            }

    def critic(self) -> bool:
        """Evaluates whether the executing state yielded a successful end goal."""
        metric = self.memory.state.get("final_metric")
        if metric is None: return False
        
        self.memory.state["status"] = AgentState.CRITIQUING.value
        self.memory.save()

        trial_results = self.memory.state.get("trial_results") or {}
        training_time = float(trial_results.get("training_time", 0.0) or 0.0)
        failures = len([a for a in self.memory.state["actions_taken"] if a["status"] == "failed"])
        budget_left = max(0, self.action_budget - len(self.memory.state["actions_taken"]))

        quality_score = float(metric >= self.metric_threshold)
        time_score = max(0.0, 1.0 - min(training_time / 120.0, 1.0))
        stability_score = 0.0 if failures > 0 else 1.0
        budget_score = min(1.0, budget_left / max(1.0, self.action_budget))
        drift_risk_score = 1.0 if self.memory.state.get("dataset_fingerprint") else 0.5
        aggregate = (0.45 * quality_score) + (0.2 * time_score) + (0.2 * stability_score) + (0.1 * budget_score) + (0.05 * drift_risk_score)
        self.memory.state["critic_breakdown"] = {
            "quality": quality_score,
            "time": time_score,
            "stability": stability_score,
            "budget": budget_score,
            "drift_risk": drift_risk_score,
            "aggregate": aggregate,
        }

        if metric >= self.metric_threshold and aggregate >= 0.55:
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
            plan = self.planner()
            if self.memory.state["status"] == AgentState.FAILED.value:
                if plan:
                    exec_result = self.executor(plan["action"], rationale=plan["rationale"])
                    if not exec_result.get("success"):
                        print(f"⚠️ [Agent] Recovery action failed: {exec_result.get('error_class')}")
                else: 
                    break # Out of options
                continue
            
            self.memory.state["status"] = AgentState.PLANNING.value
            plan = self.planner()
            
            if not plan:
                # All primary execution steps finished, run critic
                success = self.critic()
                if success:
                    break
            else:
                exec_result = self.executor(plan["action"], rationale=plan["rationale"])
                if not exec_result.get("success"):
                    print(f"⚠️ [Agent] Action {plan['action'].value} failed. Halting workflow.")
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
    parser.add_argument("--action-budget", type=int, default=12, help="Maximum number of actions allowed in a run")
    args = parser.parse_args()

    agent = MetaTuneAgent(
        data_path=args.data, 
        target_col=args.target, 
        metric_threshold=args.threshold,
        use_bilevel=args.bilevel,
        force_new=args.new,
        approval_mode=args.approval_mode,
        action_budget=args.action_budget
    )
    agent.run()

if __name__ == "__main__":
    main()
