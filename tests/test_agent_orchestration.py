import json
import os
import tempfile
import unittest
from unittest.mock import patch
import sys
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


if __name__ == "__main__":
    unittest.main()
