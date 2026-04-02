
import unittest
import pandas as pd
import numpy as np
import os
import sys
import io
from contextlib import redirect_stdout
from data_analyzer import DatasetAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV for testing
        self.test_csv = "test_data_analyzer_dummy.csv"
        df = pd.DataFrame({
            'A': np.random.rand(10),
            'B': np.random.choice(['x', 'y'], 10),
            'target': np.random.choice([0, 1], 10)
        })
        df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_analyzer_functionality(self):
        # Determine strict functionality without printing
        analyzer = DatasetAnalyzer(self.test_csv, target_col='target')
        self.assertTrue(analyzer.load_data())
        dna = analyzer.analyze()
        self.assertIsNotNone(dna)
        self.assertIn('n_instances', dna)
        self.assertEqual(dna['n_instances'], 10)
        
    def test_cli_behavior_simulation(self):
        # mimic the old main block logic
        f = io.StringIO()
        with redirect_stdout(f):
            analyzer = DatasetAnalyzer(self.test_csv, target_col='target')
            if analyzer.load_data():
                dna = analyzer.analyze()
                analyzer.print_summary()
                print("\n✅ Analysis Complete. DNA ready for Meta-Brain.")
        
        output = f.getvalue()
        self.assertIn("DATASET DNA", output)
        self.assertIn("Analysis Complete", output)

    def test_auto_detect_target_not_last_column(self):
        test_csv = "test_target_middle.csv"
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "target_label": [0, 1, 0, 1],
            "feature2": [10, 11, 12, 13],
        })
        df.to_csv(test_csv, index=False)
        try:
            analyzer = DatasetAnalyzer(test_csv, target_col=None)
            self.assertTrue(analyzer.load_data())
            dna = analyzer.analyze()
            self.assertIsNotNone(dna)
            self.assertEqual(analyzer.target_col, "target_label")
            self.assertIn("target_detection_method", dna)
        finally:
            if os.path.exists(test_csv):
                os.remove(test_csv)

    def test_cleaning_handles_null_values(self):
        test_csv = "test_null_cleaning.csv"
        df = pd.DataFrame({
            "A": [1.0, np.nan, 3.0, np.nan],
            "B": ["x", None, "y", None],
            "target": [1, 0, 1, 0],
        })
        df.to_csv(test_csv, index=False)
        try:
            analyzer = DatasetAnalyzer(test_csv, target_col="target")
            self.assertTrue(analyzer.load_data())
            dna = analyzer.analyze()
            self.assertIsNotNone(dna)
            self.assertIn("cleaning_report", dna)
            self.assertEqual(int(analyzer.cleaned_data.isnull().sum().sum()), 0)
        finally:
            if os.path.exists(test_csv):
                os.remove(test_csv)

if __name__ == '__main__':
    unittest.main()
