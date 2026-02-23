import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vizier_stub import Study, Trial, SearchSpace, Measurement

class TestVizierStub(unittest.TestCase):

    def test_study_trials_no_collision(self):
        study = Study(name="test")
        self.assertIsInstance(study.trials, list)
        self.assertEqual(len(study.trials), 0)

    def test_study_add_and_get_trial(self):
        study = Study(name="test")
        trial = Trial(id=0, parameters={"lr": 0.01})
        trial.complete(0.95, elapsed_secs=1.5)
        study.add_trial(trial)
        self.assertEqual(len(study.get_trials()), 1)
        self.assertEqual(study.get_trials()[0].final_measurement, 0.95)

    def test_trial_status(self):
        trial = Trial(id=0, parameters={"lr": 0.01})
        self.assertEqual(trial.status, "ACTIVE")
        trial.complete(0.9)
        self.assertEqual(trial.status, "COMPLETED")

    def test_optimal_trials(self):
        study = Study(name="test")
        for i, score in enumerate([0.7, 0.95, 0.8]):
            t = Trial(id=i, parameters={"lr": 0.01 * (i+1)})
            t.complete(score)
            study.add_trial(t)
        best = study.optimal_trials()
        self.assertAlmostEqual(best[0].final_measurement, 0.95)

    def test_search_space_validate(self):
        class _Config:
            def __init__(self, bounds):
                self.bounds = bounds
        
        space = SearchSpace()
        space.parameters["lr"] = _Config([1e-4, 1e-1])
        self.assertTrue(space.validate({"lr": 0.01}))
        self.assertFalse(space.validate({"lr": 10.0}))  # out of bounds

    def test_measurement_creation(self):
        m = Measurement(metrics={"accuracy": 0.95, "loss": 0.1}, elapsed_secs=2.5, step=100)
        self.assertAlmostEqual(m.metrics["accuracy"], 0.95)

if __name__ == '__main__':
    unittest.main()
