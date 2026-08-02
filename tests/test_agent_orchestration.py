import json
import os
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import EpisodicMemory, MetaTuneAgent, ActionType, FailureType


class TestAgentOrchestration(unittest.TestCase):
    def test_force_new_resets_memory_to_idle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_file = os.path.join(tmpdir, "episodic_memory.json")
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": "old-run",
                    "status": "completed",
                    "actions_taken": [{"action": "inspect_dataset", "status": "success"}],
                }, f)

            mem = EpisodicMemory(memory_file=memory_file)
            mem.load(force_new=True)

            self.assertEqual(mem.state["status"], "idle")
            self.assertEqual(mem.state["actions_taken"], [])
            self.assertNotEqual(mem.state["run_id"], "old-run")

    def test_request_approval_non_interactive_manual_uses_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")

            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="manual",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )

            with patch("sys.stdin.isatty", return_value=False):
                self.assertFalse(agent.request_approval("Should default to no", default_response=False))
                self.assertTrue(agent.request_approval("Should default to yes", default_response=True))

    def test_full_auto_approval_always_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")

            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            self.assertTrue(agent.request_approval("Any gate"))

    def test_structured_action_logging_includes_failure_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_file = os.path.join(tmpdir, "episodic_memory.json")
            mem = EpisodicMemory(memory_file=memory_file)

            mem.log_action(
                ActionType.INSPECT_DATASET,
                "failed",
                {"message": "bad column"},
                failure_type=FailureType.DATA,
            )

            event = mem.state["actions_taken"][-1]
            self.assertEqual(event["failure_type"], FailureType.DATA.value)
            self.assertIn("details", event)
            self.assertEqual(event["details"]["message"], "bad column")

    def test_policy_planner_returns_action_and_rationale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")

            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            plan = agent.planner()
            self.assertIsNotNone(plan)
            self.assertEqual(plan["action"], ActionType.INSPECT_DATASET)
            self.assertIn("rationale", plan)

    def test_critic_records_multi_objective_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")

            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            agent.memory.state["final_metric"] = 0.9
            agent.memory.state["trial_results"] = {"training_time": 2.0}
            ok = agent.critic()
            self.assertTrue(ok)
            self.assertIn("critic_breakdown", agent.memory.state)
            self.assertIn("aggregate", agent.memory.state["critic_breakdown"])

    def test_executor_records_action_costs_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")

            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            result = agent.executor(ActionType.INSPECT_DATASET, rationale="unit-test")
            self.assertTrue(result["success"])
            self.assertGreaterEqual(result["confidence"], 0.05)
            self.assertLessEqual(result["confidence"], 0.99)
            self.assertIn("cost_summary", result["cost"])
            self.assertGreaterEqual(agent.memory.state["cost_summary"]["total_actions"], 1)
            self.assertGreaterEqual(len(agent.memory.state["action_costs"]), 1)

    def test_tool_registry_has_core_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            self.assertIn(ActionType.INSPECT_DATASET, agent.tool_registry)
            self.assertIn(ActionType.RUN_TRIAL, agent.tool_registry)

    def test_guardrail_blocks_when_max_trials_reached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
                max_trials=1,
            )
            agent.memory.state["actions_taken"].append({"action": "run_trial", "status": "success"})
            reason = agent._guardrails_allow(ActionType.RUN_TRIAL)
            self.assertIsNotNone(reason)
            self.assertIn("max_trials", reason)

    def test_memory_architecture_working_episodic_semantic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_file = os.path.join(tmpdir, "episodic_memory.json")
            mem = EpisodicMemory(memory_file=memory_file)
            dna = {
                "task_type": "classification",
                "n_features": 24,
                "n_instances": 4200,
                "sparsity": 0.2,
            }
            mem.set_working_memory("dataset_profile", {"task_type": "classification"})
            mem.add_episode("inspect_dataset", {"dna_feature_count": 10})
            mem.update_semantic_memory(dna=dna, metric=0.81, params={"learning_rate": 0.01})
            retrieved = mem.retrieve_semantic_context(dna=dna, top_k=1)

            self.assertIn("dataset_profile", mem.state["working_memory"])
            self.assertGreaterEqual(len(mem.state["episodic_memory"]), 1)
            self.assertGreaterEqual(len(mem.state["semantic_memory"]), 1)
            self.assertEqual(len(retrieved), 1)
            self.assertIn("score", retrieved[0])

    def test_postmortem_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_file = os.path.join(tmpdir, "episodic_memory.json")
            mem = EpisodicMemory(memory_file=memory_file)
            mem.add_postmortem(
                action=ActionType.RUN_TRIAL,
                failure_type=FailureType.TRAINING,
                message="metric dropped",
                tags=["training_error", "regression"],
            )
            self.assertEqual(len(mem.state["postmortems"]), 1)
            self.assertEqual(mem.state["postmortems"][0]["failure_type"], FailureType.TRAINING.value)

    def test_preflight_checks_detect_blockers_and_issues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            report = agent._run_preflight_checks({
                "task_type": "classification",
                "n_features": 0,
                "missing_ratio": 0.75,
                "class_imbalance_ratio": 50,
            })
            self.assertFalse(report["ok"])
            self.assertIn("no_features", report["blockers"])
            self.assertIn("high_missing_ratio", report["issues"])
            self.assertIn("high_class_imbalance", report["issues"])

    def test_planner_writes_machine_readable_decision_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            plan = agent.planner()
            self.assertIsNotNone(plan)
            self.assertGreaterEqual(len(agent.memory.state["decision_traces"]), 1)
            last_trace = agent.memory.state["decision_traces"][-1]
            self.assertEqual(last_trace["stage"], "planner")
            self.assertEqual(last_trace["decision"], "selected")

    def test_low_confidence_abstention_on_run_trial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            agent.memory.state["dataset_dna"] = {
                "task_type": "classification",
                "missing_ratio": 0.95,
                "sparsity": 0.95,
            }
            agent.memory.state["working_memory"]["semantic_context"] = []
            result = agent.executor(ActionType.RUN_TRIAL, rationale="abstention-test")
            self.assertFalse(result["success"])
            self.assertEqual(result["error_class"], "abstained_low_confidence")
            self.assertGreaterEqual(len(agent.memory.state["abstentions"]), 1)

    def test_preflight_blocks_on_leakage_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            report = agent._run_preflight_checks(
                {"task_type": "classification", "n_features": 2, "missing_ratio": 0.0},
                leakage_signals=["feature_equals_target:leaky_col"],
            )
            self.assertFalse(report["ok"])
            self.assertIn("data_leakage_suspected", report["blockers"])
            self.assertIn("feature_equals_target:leaky_col", report["issues"])

    def test_detect_data_leakage_signals_from_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            df = pd.DataFrame({
                "feature_a": [1, 2, 3, 4, 5, 6],
                "leaky_target": [0, 1, 0, 1, 0, 1],
                "target": [0, 1, 0, 1, 0, 1],
            })
            signals = agent._detect_data_leakage_signals(df, target_col="target", task_type="classification")
            self.assertTrue(any("feature_equals_target" in s for s in signals) or any("suspicious_feature_name" in s for s in signals))


def _write_synthetic_classification_csv(path, target_col="target", n_rows=40, seed=0):
    """A small but learnable binary classification CSV, sized similarly to
    tests/test_pipeline.py's fixture — enough rows for a real
    train_test_split + a few epochs of actual PyTorch training to run
    quickly without being degenerate."""
    rng = np.random.RandomState(seed)
    feature_a = rng.rand(n_rows)
    feature_b = rng.rand(n_rows)
    df = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
        target_col: (feature_a + feature_b > 1.0).astype(int),
    })
    df.to_csv(path, index=False)


class TestAgentRecoveryLoop(unittest.TestCase):
    """Regression tests for the diagnose -> revise -> retry loop, target-
    column auto-detection propagation, and cleaned-data reuse. Each of
    these covers a bug that was previously invisible to the test suite
    precisely because nothing exercised agent.run() end-to-end or checked
    what actually reached the trainer."""

    def test_run_retries_below_threshold_trials_and_terminates_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            _write_synthetic_classification_csv(data_path)

            agent = MetaTuneAgent(
                data_path=data_path,
                target_col="target",
                metric_threshold=1.01,  # unreachable — every trial must "fail" the critic
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
                output_dir=os.path.join(tmpdir, "runs"),
                action_budget=30,
                max_trials=2,
            )
            agent.run()

            # The run must not falsely report success just because a
            # revision was computed.
            self.assertEqual(agent.memory.state["status"], "failed")

            # It must have actually retried — not stopped after trial 1.
            run_trial_attempts = [a for a in agent.memory.state["actions_taken"] if a["action"] == ActionType.RUN_TRIAL.value]
            self.assertEqual(len(run_trial_attempts), 2, "expected exactly max_trials RUN_TRIAL attempts to be logged")
            self.assertTrue(all(a["status"] == "success" for a in run_trial_attempts), "each trial should complete and be judged by the critic, not crash")

            # revise_strategy must have actually run and changed something,
            # not been skipped entirely.
            self.assertGreaterEqual(agent.memory.state.get("revision_count", 0), 1)

            # Termination must be the max_trials guardrail, not an
            # unrelated/ambiguous stop.
            self.assertIn("max_trials", agent.memory.state.get("failure_reason", ""))

    def test_run_completes_in_one_trial_when_threshold_is_reachable(self):
        """Happy-path sanity check: the retry machinery must not fire (and
        must not cost extra trials) when the very first trial already
        clears the threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            _write_synthetic_classification_csv(data_path)

            agent = MetaTuneAgent(
                data_path=data_path,
                target_col="target",
                metric_threshold=0.0,  # trivially reachable
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
                output_dir=os.path.join(tmpdir, "runs"),
                action_budget=30,
                max_trials=3,
            )
            agent.run()

            self.assertEqual(agent.memory.state["status"], "completed")
            run_trial_attempts = [a for a in agent.memory.state["actions_taken"] if a["action"] == ActionType.RUN_TRIAL.value]
            self.assertEqual(len(run_trial_attempts), 1)
            self.assertEqual(agent.memory.state.get("revision_count", 0), 0)

    def test_target_col_autodetect_propagates_from_analyzer_to_agent(self):
        """The target column the agent actually resolves to must match what
        DatasetAnalyzer detected — not silently fall back to "last column"
        downstream in the trainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            # 'target' sits in the MIDDLE of the columns on purpose: a
            # naive "last column" fallback would pick feature_b instead.
            rng = np.random.RandomState(0)
            df = pd.DataFrame({
                "feature_a": rng.rand(40),
                "target": rng.randint(0, 2, 40),
                "feature_b": rng.rand(40),
            })
            df.to_csv(data_path, index=False)

            agent = MetaTuneAgent(
                data_path=data_path,
                target_col=None,  # force auto-detection
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
                output_dir=os.path.join(tmpdir, "runs"),
            )
            self.assertIsNone(agent.target_col)
            agent._tool_inspect_dataset()

            self.assertEqual(agent.target_col, "target")
            self.assertIsNotNone(agent._cleaned_data_cache)
            self.assertIn("target", agent._cleaned_data_cache.columns)

    def test_run_trial_reuses_cached_data_after_source_csv_removed(self):
        """RUN_TRIAL must train on the DataFrame INSPECT_DATASET already
        cleaned and cached, not re-read data_path from disk — proven here
        by deleting the CSV in between and confirming the trial still
        succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            _write_synthetic_classification_csv(data_path)

            agent = MetaTuneAgent(
                data_path=data_path,
                target_col="target",
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
                output_dir=os.path.join(tmpdir, "runs"),
            )
            agent._tool_inspect_dataset()
            agent._tool_propose_search_space()

            os.remove(data_path)  # the trainer must not need this anymore

            details = agent._tool_run_trial()
            self.assertIn("final_metric", agent.memory.state)
            self.assertIsNotNone(agent.memory.state["final_metric"])

    def test_planner_appends_exactly_one_decision_trace_per_call(self):
        """Direct guard against the double-invocation bug: a single
        planner() call must append exactly one decision trace, so a caller
        that (bug-for-bug) invokes it twice per loop iteration would be
        caught by an integration-level trace-count check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.csv")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x,target\n1,0\n2,1\n")
            agent = MetaTuneAgent(
                data_path=data_path,
                approval_mode="full-auto",
                force_new=True,
                memory_file=os.path.join(tmpdir, "episodic_memory.json"),
            )
            before = len(agent.memory.state["decision_traces"])
            agent.planner()
            after = len(agent.memory.state["decision_traces"])
            self.assertEqual(after - before, 1)


if __name__ == "__main__":
    unittest.main()
