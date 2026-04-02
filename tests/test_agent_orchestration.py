import json
import os
import tempfile
import unittest
from unittest.mock import patch
import sys

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


if __name__ == "__main__":
    unittest.main()
