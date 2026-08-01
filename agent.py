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

import pandas as pd

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

@dataclass
class ToolSpec:
    action: ActionType
    handler_name: str
    retry_by_error: Dict[str, int]

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
            "action_costs": [],
            "working_memory": {},
            "episodic_memory": [],
            "semantic_memory": {},
            "postmortems": [],
            "decision_traces": [],
            "preflight_checks": [],
            "abstentions": [],
            "trial_results": None,
            "final_metric": None,
            "status": AgentState.IDLE.value,
            "failure_reason": None,
            "cost_summary": {
                "total_actions": 0,
                "total_runtime_sec": 0.0,
                "total_cost_units": 0.0,
            },
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

    def set_working_memory(self, key: str, value: Any):
        self.state["working_memory"][key] = value
        self.save()

    def add_episode(self, episode_type: str, payload: Dict[str, Any]):
        self.state["episodic_memory"].append({
            "type": episode_type,
            "payload": payload,
            "timestamp": time.time(),
        })
        self.save()

    def _semantic_key_from_dna(self, dna: Dict[str, Any]) -> str:
        task = dna.get("task_type", "unknown")
        n_feat_bucket = int(float(dna.get("n_features", 0)) // 10)
        sparsity_bucket = round(float(dna.get("sparsity", 0.0)), 1)
        n_inst_bucket = int(float(dna.get("n_instances", 0)) // 1000)
        return f"{task}|f{n_feat_bucket}|s{sparsity_bucket}|n{n_inst_bucket}"

    def update_semantic_memory(self, dna: Dict[str, Any], metric: float, params: Optional[Dict[str, Any]] = None):
        key = self._semantic_key_from_dna(dna)
        node = self.state["semantic_memory"].get(key, {
            "key": key,
            "task_type": dna.get("task_type", "unknown"),
            "run_count": 0,
            "avg_metric": 0.0,
            "last_metric": 0.0,
            "last_params": {},
            "last_updated": 0.0,
        })
        run_count = int(node["run_count"]) + 1
        prev_avg = float(node.get("avg_metric", 0.0))
        node["avg_metric"] = ((prev_avg * (run_count - 1)) + float(metric)) / run_count
        node["run_count"] = run_count
        node["last_metric"] = float(metric)
        node["last_params"] = params or node.get("last_params", {})
        node["last_updated"] = time.time()
        self.state["semantic_memory"][key] = node
        self.save()

    def retrieve_semantic_context(self, dna: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.state["semantic_memory"]:
            return []
        task = dna.get("task_type", "unknown")
        n_features = float(dna.get("n_features", 0.0))
        n_instances = float(dna.get("n_instances", 0.0))
        sparsity = float(dna.get("sparsity", 0.0))
        scored = []
        for node in self.state["semantic_memory"].values():
            task_score = 1.0 if node.get("task_type") == task else 0.3
            key_parts = node["key"].split("|")
            try:
                f_bucket = float(key_parts[1].lstrip("f")) * 10.0
                s_bucket = float(key_parts[2].lstrip("s"))
                n_bucket = float(key_parts[3].lstrip("n")) * 1000.0
            except Exception:
                f_bucket, s_bucket, n_bucket = 0.0, 0.0, 0.0
            feat_score = 1.0 / (1.0 + abs(n_features - f_bucket) / 50.0)
            inst_score = 1.0 / (1.0 + abs(n_instances - n_bucket) / 5000.0)
            sparse_score = 1.0 / (1.0 + abs(sparsity - s_bucket) / 0.5)
            metric_score = max(0.0, min(1.0, float(node.get("avg_metric", 0.0))))
            final = (0.35 * task_score) + (0.2 * feat_score) + (0.2 * inst_score) + (0.15 * sparse_score) + (0.1 * metric_score)
            scored.append((final, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": round(score, 4), **node} for score, node in scored[:top_k]]

    def add_postmortem(self, action: ActionType, failure_type: FailureType, message: str, tags: List[str]):
        self.state["postmortems"].append({
            "action": action.value,
            "failure_type": failure_type.value,
            "message": message,
            "tags": tags,
            "timestamp": time.time(),
        })
        self.save()

    def add_decision_trace(self, stage: str, action: str, decision: str, rationale: str, confidence: Optional[float] = None):
        self.state["decision_traces"].append({
            "stage": stage,
            "action": action,
            "decision": decision,
            "rationale": rationale,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        self.save()

    def add_preflight_check(self, report: Dict[str, Any]):
        self.state["preflight_checks"].append({
            **report,
            "timestamp": time.time(),
        })
        self.save()

    def add_abstention(self, action: ActionType, reason: str, confidence: float):
        self.state["abstentions"].append({
            "action": action.value,
            "reason": reason,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        self.save()

# ==========================================
# 3. Core Agent (Planner, Executor, Critic)
# ==========================================

class MetaTuneAgent:
    def __init__(self, data_path: str, target_col: Optional[str] = None, metric_threshold: float = 0.0, use_bilevel: bool = False, force_new: bool = False, approval_mode: str = "manual", memory_file: Optional[str] = None, output_dir: str = ".metatune_runs", action_budget: int = 12, max_runtime_sec: int = 900, max_trials: int = 3):
        self.data_path = data_path
        self.target_col = target_col
        self.metric_threshold = metric_threshold
        self.use_bilevel = use_bilevel
        self.approval_mode = approval_mode
        self.action_budget = action_budget
        self.max_runtime_sec = max_runtime_sec
        self.max_trials = max_trials
        self.run_started_at = time.time()

        # All generated run artifacts (episodic memory, run reports) live
        # under output_dir by default rather than the current working
        # directory, so running the agent from a repo checkout doesn't leave
        # episodic_memory.json / agent_run_report_*.json sitting in the repo.
        # Callers that pass an explicit memory_file (e.g. tests pointing at a
        # tmpdir) get exactly that path, unchanged.
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if memory_file is None:
            memory_file = os.path.join(self.output_dir, "episodic_memory.json")

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
        self.tool_registry = self._init_tool_registry()

    def _init_task_graph(self) -> Dict[ActionType, TaskNode]:
        return {
            ActionType.INSPECT_DATASET: TaskNode(ActionType.INSPECT_DATASET, [], deadline_epoch=2, retries_left=2),
            ActionType.PROPOSE_SEARCH_SPACE: TaskNode(ActionType.PROPOSE_SEARCH_SPACE, [ActionType.INSPECT_DATASET], deadline_epoch=5, retries_left=2),
            ActionType.RUN_TRIAL: TaskNode(ActionType.RUN_TRIAL, [ActionType.PROPOSE_SEARCH_SPACE], deadline_epoch=10, retries_left=2),
            ActionType.DIAGNOSE_FAILURE: TaskNode(ActionType.DIAGNOSE_FAILURE, [ActionType.RUN_TRIAL], deadline_epoch=11, retries_left=1),
            ActionType.REVISE_STRATEGY: TaskNode(ActionType.REVISE_STRATEGY, [ActionType.DIAGNOSE_FAILURE], deadline_epoch=12, retries_left=1),
        }

    def _init_tool_registry(self) -> Dict[ActionType, ToolSpec]:
        return {
            ActionType.INSPECT_DATASET: ToolSpec(
                action=ActionType.INSPECT_DATASET,
                handler_name="_tool_inspect_dataset",
                retry_by_error={FailureType.IO.value: 1, FailureType.DATA.value: 0, FailureType.UNKNOWN.value: 1},
            ),
            ActionType.PROPOSE_SEARCH_SPACE: ToolSpec(
                action=ActionType.PROPOSE_SEARCH_SPACE,
                handler_name="_tool_propose_search_space",
                retry_by_error={FailureType.IO.value: 1, FailureType.DATA.value: 1, FailureType.UNKNOWN.value: 1},
            ),
            ActionType.RUN_TRIAL: ToolSpec(
                action=ActionType.RUN_TRIAL,
                handler_name="_tool_run_trial",
                retry_by_error={FailureType.TRAINING.value: 1, FailureType.IO.value: 1, FailureType.UNKNOWN.value: 1},
            ),
            ActionType.DIAGNOSE_FAILURE: ToolSpec(
                action=ActionType.DIAGNOSE_FAILURE,
                handler_name="_tool_diagnose_failure",
                retry_by_error={FailureType.UNKNOWN.value: 0},
            ),
            ActionType.REVISE_STRATEGY: ToolSpec(
                action=ActionType.REVISE_STRATEGY,
                handler_name="_tool_revise_strategy",
                retry_by_error={FailureType.UNKNOWN.value: 0},
            ),
        }

    def _guardrails_allow(self, action: ActionType) -> Optional[str]:
        elapsed = time.time() - self.run_started_at
        if elapsed > self.max_runtime_sec:
            return f"Guardrail max_runtime_sec={self.max_runtime_sec} exceeded."
        if action == ActionType.RUN_TRIAL:
            run_trial_actions = [a for a in self.memory.state.get("actions_taken", []) if a.get("action") == ActionType.RUN_TRIAL.value]
            if len(run_trial_actions) >= self.max_trials:
                return f"Guardrail max_trials={self.max_trials} exceeded."
        return None

    def _tool_inspect_dataset(self) -> Dict[str, Any]:
        analyzer = DatasetAnalyzer(self.data_path, target_col=self.target_col)
        if not analyzer.load_data():
            raise ValueError("Failed to load dataset for analysis.")
        dna = analyzer.analyze()
        if dna is None:
            raise ValueError("Dataset analysis failed: DNA signature could not be extracted (possibly due to invalid target column or entirely null data).")
        self.memory.state["dataset_dna"] = dna
        semantic_context = self.memory.retrieve_semantic_context(dna, top_k=3)
        self.memory.set_working_memory("semantic_context", semantic_context)
        self.memory.set_working_memory("dataset_profile", {
            "task_type": dna.get("task_type"),
            "n_features": dna.get("n_features"),
            "n_instances": dna.get("n_instances"),
            "sparsity": dna.get("sparsity"),
        })
        details = {
            "message": "Dataset analysis completed",
            "dna_feature_count": len(dna),
            "semantic_hits": len(semantic_context),
        }
        self.memory.log_action(ActionType.INSPECT_DATASET, "success", details)
        self.memory.add_episode("inspect_dataset", details)
        leakage_signals = self._detect_data_leakage_signals(
            data=getattr(analyzer, "cleaned_data", None),
            target_col=getattr(analyzer, "target_col", None),
            task_type=dna.get("task_type"),
        )
        preflight = self._run_preflight_checks(dna, leakage_signals=leakage_signals)
        self.memory.add_preflight_check(preflight)
        self.memory.add_decision_trace(
            stage="preflight",
            action=ActionType.INSPECT_DATASET.value,
            decision="pass" if preflight["ok"] else "block",
            rationale=preflight["summary"],
            confidence=1.0 if preflight["ok"] else 0.2,
        )
        if not preflight["ok"]:
            raise ValueError(f"Preflight checks failed: {preflight['summary']}")
        return details

    def _tool_propose_search_space(self) -> Dict[str, Any]:
        dna = self.memory.state.get("dataset_dna")
        params = self.meta_learner.predict(dna)
        algos = algorithm_recommender.recommend_algorithms(dna)
        self.memory.state["predicted_params"] = params
        self.memory.state["recommended_algorithms"] = algos

        if getattr(self.meta_learner, "is_trained", False) is False and os.path.exists(getattr(self.meta_learner, "knowledge_base_path", "knowledge_base.csv")):
            if self.request_approval("MetaBrain has historical data. Train MetaBrain prior to predicting?", auto_approve=False, default_response=False):
                self.meta_learner.train()

        details = {
            "message": "Search space proposed",
            "predicted_params": params,
            "recommended_algorithms": algos,
        }
        self.memory.log_action(ActionType.PROPOSE_SEARCH_SPACE, "success", details)
        self.memory.set_working_memory("latest_params", params)
        self.memory.add_episode("propose_search_space", {"n_algorithms": len(algos.get("recommendations", [])) if isinstance(algos, dict) else 0})
        return details

    def _tool_run_trial(self) -> Dict[str, Any]:
        # A fresh trial is starting: give the recovery machinery (diagnose /
        # revise) a clean slate so it can react to THIS trial's outcome even
        # if it already ran once earlier in this run.
        self.task_graph[ActionType.DIAGNOSE_FAILURE].status = TaskStatus.PENDING
        self.task_graph[ActionType.REVISE_STRATEGY].status = TaskStatus.PENDING

        params = self.memory.state.get("predicted_params")
        dna = self.memory.state.get("dataset_dna")

        if self.use_bilevel:
            if self.request_approval(f"Proceed with Expensive Bilevel Optimization (N=trials, Evolutionary tuning)?", auto_approve=False, default_response=False):
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

        previous_metric = self.memory.state.get("previous_metric")
        current_metric = float(results.get("final_metric", 0.0))
        rollback_recommended = previous_metric is not None and (current_metric + 1e-9) < float(previous_metric)
        self.memory.state["previous_metric"] = current_metric

        details = {
            "message": "Training trial completed",
            "final_metric": current_metric,
            "metric_name": results.get("metric_name"),
            "rollback_recommended": rollback_recommended,
        }
        self.memory.log_action(ActionType.RUN_TRIAL, "success", details)
        self.memory.add_episode("run_trial", details)
        self.memory.update_semantic_memory(dna=dna, metric=current_metric, params=params)
        self.memory.set_working_memory("latest_metric", current_metric)

        if self.request_approval("Store trial experience logically back to Knowledge Base?", auto_approve=True, default_response=True):
             self.meta_learner.store_experience(dna, params, self.memory.state["final_metric"])
        return details

    def _tool_diagnose_failure(self) -> Dict[str, Any]:
        metric = self.memory.state.get("final_metric")
        if metric is None:
            # RUN_TRIAL never produced a metric, so this is a hard execution
            # failure (exception), not a below-threshold result. Report the
            # real cause instead of a fabricated "metric fell below
            # threshold" message that would misrepresent what happened.
            underlying = self.memory.state.get("failure_reason") or "no metric was recorded and no error message was captured."
            failure = f"Trial execution failed: {underlying}"
            diagnosis_kind = "execution_error"
        else:
            failure = f"Metric ({metric}) fell below threshold ({self.metric_threshold})."
            diagnosis_kind = "threshold_miss"

        self.memory.state["failure_reason"] = failure
        self.memory.state["diagnosis_kind"] = diagnosis_kind
        print(f"   Diagnosis: {failure}")
        details = {"message": "Failure diagnosed", "reason": failure, "diagnosis_kind": diagnosis_kind}
        self.memory.log_action(ActionType.DIAGNOSE_FAILURE, "success", details, failure_type=FailureType.TRAINING)
        self.memory.add_episode("diagnose_failure", details)
        # executor() unconditionally marks status "executing" while a tool
        # runs; explicitly restore FAILED here (mirroring how
        # _tool_revise_strategy controls status at the end of its own turn)
        # so run()'s recovery branch stays engaged for the follow-up
        # REVISE_STRATEGY step instead of falling through to a redundant
        # critic() re-evaluation of the same stale metric.
        self.memory.state["status"] = AgentState.FAILED.value
        return details

    def _tool_revise_strategy(self) -> Dict[str, Any]:
        revision_count = int(self.memory.state.get("revision_count", 0)) + 1
        self.memory.state["revision_count"] = revision_count

        params = dict(self.memory.state.get("predicted_params") or {})
        changes = []
        if params.get('learning_rate'):
            params['learning_rate'] = max(float(params['learning_rate']) / 2.0, 1e-6)
            changes.append(f"learning_rate -> {params['learning_rate']:.6g}")
        if 'dropout' in params:
            params['dropout'] = float(min(float(params.get('dropout', 0.1)) + 0.05, 0.5))
            changes.append(f"dropout -> {params['dropout']:.3f}")
        self.memory.state["predicted_params"] = params

        revision_summary = ", ".join(changes) if changes else "no tunable params available to revise"
        print(f"   [Revise] Attempt #{revision_count}: {revision_summary}. Re-queuing a trial with the updated params.")

        details = {
            "message": "Strategy revised",
            "revision": revision_summary,
            "revision_count": revision_count,
            "updated_params": params,
        }
        self.memory.log_action(ActionType.REVISE_STRATEGY, "success", details)
        self.memory.add_episode("revise_strategy", details)

        # This is the crux of self-correction: hand control back to the
        # planner instead of declaring the run COMPLETED here. Resetting
        # RUN_TRIAL to PENDING makes it a valid candidate again (its only
        # dependency, PROPOSE_SEARCH_SPACE, is already satisfied), so the
        # next planning pass will re-run the trial with the revised params.
        # The RUN_TRIAL guardrail (--max-trials) is what actually bounds how
        # many times this can happen, so looping back here is always safe.
        self.task_graph[ActionType.RUN_TRIAL].status = TaskStatus.PENDING
        self.memory.state["status"] = AgentState.PLANNING.value
        return details

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
        """Classify a caught exception for retry-policy and postmortem
        purposes. Exception type is checked first since it's a reliable
        signal; message-keyword matching is a fallback for the generic
        ValueError/RuntimeError types that pandas/numpy/torch tend to raise
        for very different underlying reasons."""
        type_map = {
            FileNotFoundError: FailureType.IO,
            PermissionError: FailureType.IO,
            IsADirectoryError: FailureType.IO,
            json.JSONDecodeError: FailureType.IO,
            KeyError: FailureType.DATA,
            pd.errors.EmptyDataError: FailureType.DATA,
            pd.errors.ParserError: FailureType.DATA,
        }
        for exc_type, failure_type in type_map.items():
            if isinstance(exc, exc_type):
                return failure_type

        msg = str(exc).lower()
        if any(k in msg for k in ["csv", "dataset", "column", "target", "data", "load", "read-only", "read only"]):
            return FailureType.DATA
        if any(k in msg for k in ["permission", "approve", "input"]):
            return FailureType.APPROVAL
        if any(k in msg for k in ["file", "path", "write", "json"]):
            return FailureType.IO
        if any(k in msg for k in ["train", "optimizer", "metric", "epoch"]):
            return FailureType.TRAINING
        return FailureType.UNKNOWN

    def _estimate_confidence(self, action: ActionType, success: bool, duration_sec: float, error_class: Optional[str] = None) -> float:
        """Calibrate confidence using data quality, runtime efficiency, and failure history."""
        dna = self.memory.state.get("dataset_dna") or {}
        missing_ratio = float(dna.get("missing_ratio", 0.0) or 0.0)
        sparsity = float(dna.get("sparsity", 0.0) or 0.0)
        prior_failures = len([a for a in self.memory.state.get("actions_taken", []) if a.get("status") == "failed"])

        base = 0.85 if success else 0.2
        quality_penalty = min(0.25, (missing_ratio * 0.2) + (sparsity * 0.15))
        duration_penalty = min(0.2, duration_sec / 120.0)
        failure_penalty = min(0.2, prior_failures * 0.05)
        error_penalty = 0.1 if error_class else 0.0
        calibration = base - quality_penalty - duration_penalty - failure_penalty - error_penalty
        return float(max(0.05, min(0.99, calibration)))

    def _record_action_cost(self, action: ActionType, duration_sec: float, success: bool, error_class: Optional[str] = None):
        """Persist action-level time and synthetic cost accounting."""
        cost_units = round((duration_sec * 0.3) + (0.0 if success else 1.0), 4)
        entry = {
            "action": action.value,
            "duration_sec": round(duration_sec, 4),
            "cost_units": cost_units,
            "success": success,
            "error_class": error_class,
            "timestamp": time.time(),
        }
        self.memory.state["action_costs"].append(entry)
        self.memory.state["cost_summary"] = {
            "total_actions": len(self.memory.state["action_costs"]),
            "total_runtime_sec": round(sum(x["duration_sec"] for x in self.memory.state["action_costs"]), 4),
            "total_cost_units": round(sum(x["cost_units"] for x in self.memory.state["action_costs"]), 4),
        }
        self.memory.save()

    def _detect_data_leakage_signals(self, data: Any, target_col: Optional[str], task_type: Optional[str]) -> List[str]:
        signals = []
        if data is None or target_col is None or target_col not in getattr(data, "columns", []):
            return signals

        feature_cols = [c for c in data.columns if c != target_col]
        leakage_name_tokens = ["target", "label", "outcome", "y_true", "ground_truth"]
        for col in feature_cols:
            col_norm = str(col).lower()
            if any(tok in col_norm for tok in leakage_name_tokens):
                signals.append(f"suspicious_feature_name:{col}")

        target = data[target_col]
        for col in feature_cols:
            series = data[col]
            try:
                if series.equals(target):
                    signals.append(f"feature_equals_target:{col}")
                    continue
            except Exception:
                pass

            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                valid = pd.concat([series, target], axis=1).dropna()
                if len(valid) >= 5:
                    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                    if corr is not None and abs(float(corr)) >= 0.995:
                        signals.append(f"near_perfect_corr:{col}")
            elif task_type == "classification":
                valid = pd.concat([series.astype(str), target.astype(str)], axis=1).dropna()
                if len(valid) >= 5:
                    match_ratio = (valid.iloc[:, 0] == valid.iloc[:, 1]).mean()
                    if float(match_ratio) >= 0.995:
                        signals.append(f"feature_matches_target_tokens:{col}")
        return sorted(set(signals))

    def _run_preflight_checks(self, dna: Dict[str, Any], leakage_signals: Optional[List[str]] = None) -> Dict[str, Any]:
        issues = []
        blockers = []
        task_type = dna.get("task_type")
        if task_type not in {"classification", "regression"}:
            blockers.append("unknown_task_type")
        if float(dna.get("n_features", 0)) <= 0:
            blockers.append("no_features")

        missing_ratio = float(dna.get("missing_ratio", 0.0) or 0.0)
        if missing_ratio > 0.6:
            issues.append("high_missing_ratio")

        if task_type == "classification":
            imbalance = float(dna.get("class_imbalance_ratio", 0.0) or 0.0)
            if imbalance > 20:
                issues.append("high_class_imbalance")

        if task_type == "classification" and self.metric_threshold > 1.0:
            issues.append("metric_threshold_out_of_range_for_classification")

        leakage_signals = leakage_signals or []
        if leakage_signals:
            blockers.append("data_leakage_suspected")
            issues.extend(leakage_signals)

        ok = len(blockers) == 0
        summary = "preflight_ok" if ok else f"blockers={','.join(blockers)}"
        return {
            "ok": ok,
            "issues": issues,
            "blockers": blockers,
            "summary": summary,
            "task_type": task_type,
            "missing_ratio": missing_ratio,
            "leakage_signals": leakage_signals,
        }

    def _should_abstain(self, action: ActionType) -> Optional[Dict[str, Any]]:
        if action != ActionType.RUN_TRIAL:
            return None
        dna = self.memory.state.get("dataset_dna") or {}
        context = self.memory.state.get("working_memory", {}).get("semantic_context", [])
        top_context_score = float(context[0]["score"]) if context else 0.0
        missing_ratio = float(dna.get("missing_ratio", 0.0) or 0.0)
        sparsity = float(dna.get("sparsity", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, 0.6 + (0.3 * top_context_score) - (0.4 * missing_ratio) - (0.3 * sparsity)))
        if confidence < 0.35:
            reason = "low_confidence_abstention: insufficient prior memory match + poor data quality"
            return {"reason": reason, "confidence": confidence}
        return None

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
        self.memory.add_decision_trace(
            stage="planner",
            action=selected.value,
            decision="selected",
            rationale=rationale,
            confidence=max(0.05, min(0.99, 1.0 - uncertainty * 0.4)),
        )
        self.memory.save()
        return {"action": selected, "rationale": rationale, "remaining_budget": remaining_budget, "uncertainty": uncertainty}

    def executor(self, action: ActionType, rationale: str = "") -> Dict[str, Any]:
        """Executes the mapped schema routines."""
        print(f"\n🤖 [Executor] Executing action: {action.value}")
        self.memory.state["status"] = AgentState.EXECUTING.value
        self.memory.save()
        self.task_graph[action].status = TaskStatus.IN_PROGRESS
        started_at = time.time()
        guardrail_reason = self._guardrails_allow(action)
        if guardrail_reason:
            duration_sec = max(0.0, time.time() - started_at)
            confidence = self._estimate_confidence(action, success=False, duration_sec=duration_sec, error_class="resource_limit")
            self._record_action_cost(action, duration_sec=duration_sec, success=False, error_class="resource_limit")
            self.memory.add_decision_trace(
                stage="guardrail",
                action=action.value,
                decision="blocked",
                rationale=guardrail_reason,
                confidence=confidence,
            )
            return {
                "success": False,
                "confidence": confidence,
                "cost": {"actions_used": len(self.memory.state["actions_taken"]), "duration_sec": round(duration_sec, 4), "cost_summary": self.memory.state.get("cost_summary", {})},
                "artifacts": {"action": action.value},
                "error_class": "resource_limit",
                "rationale": f"{rationale} | {guardrail_reason}",
            }

        spec = self.tool_registry.get(action)
        if spec is None:
            raise ValueError(f"No ToolRegistry entry for action: {action.value}")

        abstention = self._should_abstain(action)
        if abstention:
            duration_sec = max(0.0, time.time() - started_at)
            self.memory.add_abstention(action, abstention["reason"], abstention["confidence"])
            self.memory.add_decision_trace(
                stage="abstention",
                action=action.value,
                decision="abstained",
                rationale=abstention["reason"],
                confidence=abstention["confidence"],
            )
            self._record_action_cost(action, duration_sec=duration_sec, success=False, error_class="abstained_low_confidence")
            return {
                "success": False,
                "confidence": abstention["confidence"],
                "cost": {
                    "actions_used": len(self.memory.state["actions_taken"]),
                    "duration_sec": round(duration_sec, 4),
                    "attempts": 0,
                    "cost_summary": self.memory.state.get("cost_summary", {}),
                },
                "artifacts": {"action": action.value},
                "error_class": "abstained_low_confidence",
                "rationale": f"{rationale} | {abstention['reason']}",
            }

        attempts = 0
        while True:
            attempts += 1
            try:
                details = getattr(self, spec.handler_name)()
                self.task_graph[action].status = TaskStatus.COMPLETED
                duration_sec = max(0.0, time.time() - started_at)
                confidence = self._estimate_confidence(action, success=True, duration_sec=duration_sec)
                self._record_action_cost(action, duration_sec=duration_sec, success=True)
                return {
                    "success": True,
                    "confidence": confidence,
                    "cost": {
                        "actions_used": len(self.memory.state["actions_taken"]),
                        "duration_sec": round(duration_sec, 4),
                        "attempts": attempts,
                        "cost_summary": self.memory.state.get("cost_summary", {}),
                    },
                    "artifacts": {"action": action.value, "details": details},
                    "error_class": None,
                    "rationale": rationale,
                }
            except Exception as e:
                failure_type = self._classify_failure(e)
                retry_limit = int(spec.retry_by_error.get(failure_type.value, 0))
                if attempts <= retry_limit:
                    continue

                self.task_graph[action].status = TaskStatus.FAILED
                self.task_graph[action].retries_left -= 1
                duration_sec = max(0.0, time.time() - started_at)
                self.memory.log_action(action, "failed", {
                    "message": str(e),
                    "exception_type": type(e).__name__,
                    "attempts": attempts,
                }, failure_type=failure_type)
                postmortem_tags = [failure_type.value, action.value, "retry_exhausted" if attempts > 1 else "single_failure"]
                self.memory.add_postmortem(action, failure_type, str(e), tags=postmortem_tags)
                self.memory.state["failure_reason"] = str(e)
                self.memory.state["status"] = AgentState.FAILED.value
                self.memory.save()
                confidence = self._estimate_confidence(action, success=False, duration_sec=duration_sec, error_class=failure_type.value)
                self._record_action_cost(action, duration_sec=duration_sec, success=False, error_class=failure_type.value)
                return {
                    "success": False,
                    "confidence": confidence,
                    "cost": {
                        "actions_used": len(self.memory.state["actions_taken"]),
                        "duration_sec": round(duration_sec, 4),
                        "attempts": attempts,
                        "cost_summary": self.memory.state.get("cost_summary", {}),
                    },
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

        # Belt-and-suspenders cap on loop iterations, independent of the
        # action_budget/max_trials guardrails. Those already bound normal
        # operation; this only guards against a future control-flow bug
        # spinning the loop without ever advancing memory.state["status"].
        max_iterations = max(50, self.action_budget * 4)
        iterations = 0

        while self.memory.state["status"] not in [AgentState.COMPLETED.value]:
            iterations += 1
            if iterations > max_iterations:
                print(f"⚠️ [Agent] Safety cap of {max_iterations} loop iterations reached. Stopping.")
                self.memory.state["status"] = AgentState.FAILED.value
                self.memory.state["failure_reason"] = f"Orchestration loop exceeded safety cap ({max_iterations} iterations)."
                self.memory.save()
                break

            is_recovering = self.memory.state["status"] == AgentState.FAILED.value
            if not is_recovering:
                self.memory.state["status"] = AgentState.PLANNING.value
            plan = self.planner()  # exactly one planner() call per iteration

            if is_recovering:
                if plan is None:
                    break  # Out of recovery options (status remains FAILED).
                exec_result = self.executor(plan["action"], rationale=plan["rationale"])
                if not exec_result.get("success"):
                    print(f"⚠️ [Agent] Recovery action failed: {exec_result.get('error_class')}")
                    # Keep looping — the next iteration re-reads status and
                    # either finds another recovery step or runs out above.
                continue

            if plan is None:
                # Nothing left in the primary plan (inspect/propose/run_trial
                # all done) — ask the critic to judge the outcome. A False
                # verdict flips status to FAILED, which the recovery branch
                # above picks up on the next iteration (diagnose -> revise ->
                # back to RUN_TRIAL with updated params, bounded by
                # --max-trials).
                if self.critic():
                    break
                continue

            exec_result = self.executor(plan["action"], rationale=plan["rationale"])
            if not exec_result.get("success"):
                error_class = exec_result.get("error_class")
                if error_class == "resource_limit":
                    # A guardrail (budget / runtime / max_trials) deliberately
                    # stopped the run. This is a clean, intentional stop, not
                    # something to route into diagnose/revise.
                    print(f"⚠️ [Agent] Guardrail stopped the run: {exec_result.get('rationale')}")
                    self.memory.state["status"] = AgentState.FAILED.value
                    self.memory.state["failure_reason"] = exec_result.get("rationale")
                else:
                    # A genuine tool failure (e.g. RUN_TRIAL raised after its
                    # own retries were exhausted). Route it through the same
                    # diagnose/revise recovery path used for below-threshold
                    # results instead of hard-aborting the whole run.
                    print(f"⚠️ [Agent] Action {plan['action'].value} failed ({error_class}). Attempting recovery.")
                    self.memory.state["status"] = AgentState.FAILED.value
                self.memory.save()

        print("\n🎉 [Agent] Flow complete. Generating final artifacts...")
        # Export logic implementation
        if self.memory.state["status"] == AgentState.COMPLETED.value:
            if self.request_approval("Export fully deployable package (.joblib/.pth)?", default_response=False):
                print("   📦 Exporting model artifacts to local directory.")
                # We would normally invoke `train_and_package` here to serialize a production artifact.
                # Simulated for the MVP hook
        else:
            print(f"   ⚠️  Run ended without meeting the success criteria: {self.memory.state.get('failure_reason', 'unknown reason')}")

        # Report
        report_path = os.path.join(self.output_dir, f"agent_run_report_{self.memory.state['run_id'][:6]}.json")
        with open(report_path, "w") as f:
            json.dump(self.memory.state, f, indent=4, cls=NumpyEncoder)
        print(f"   📄 Report written to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="MetaTune Agent orchestrator")
    parser.add_argument("data", help="Path to raw CSV dataset")
    parser.add_argument("--target", help="Target column name (optional)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum acceptable metric (Critic gate)")
    parser.add_argument("--bilevel", action="store_true", help="Launch bilevel optimizer search space")
    parser.add_argument("--new", action="store_true", help="Force ignore past episodic memory and start fresh")
    parser.add_argument("--approval-mode", choices=["manual", "semi-auto", "full-auto"], default="manual", help="Approval policy: manual prompts, semi-auto defaults, or full-auto allow")
    parser.add_argument("--action-budget", type=int, default=12, help="Maximum number of actions allowed in a run")
    parser.add_argument("--max-runtime-sec", type=int, default=900, help="Guardrail: max wall-clock runtime in seconds")
    parser.add_argument("--max-trials", type=int, default=3, help="Guardrail: max RUN_TRIAL actions in one run")
    parser.add_argument("--output-dir", default=".metatune_runs", help="Directory for episodic memory + run reports (default: .metatune_runs)")
    parser.add_argument("--memory-file", default=None, help="Override the episodic memory file path (default: <output-dir>/episodic_memory.json)")
    args = parser.parse_args()

    agent = MetaTuneAgent(
        data_path=args.data, 
        target_col=args.target, 
        metric_threshold=args.threshold,
        use_bilevel=args.bilevel,
        force_new=args.new,
        approval_mode=args.approval_mode,
        action_budget=args.action_budget,
        max_runtime_sec=args.max_runtime_sec,
        max_trials=args.max_trials,
        output_dir=args.output_dir,
        memory_file=args.memory_file,
    )
    agent.run()

if __name__ == "__main__":
    main()
