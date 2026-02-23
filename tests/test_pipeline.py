
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
from pipeline import MetaTunePipeline
import argparse

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.test_csv = "test_pipeline_dummy.csv"
        df = pd.DataFrame({
            'A': np.random.rand(20),
            'target': np.random.randint(0, 2, 20)
        })
        df.to_csv(self.test_csv, index=False)
        df.to_csv("temp.csv", index=False)
        
    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists("temp.csv"):
            os.remove("temp.csv")
            
        import glob
        for f in glob.glob("metatune_report*.json") + glob.glob("knowledge_base*.csv") + \
                  glob.glob("preprocessing_pipeline*.pkl") + glob.glob("meta_brain_weights*.pth"):
            try: os.remove(f)
            except: pass
            
    def test_pipeline_class(self):
        # Verify direct class usage
        pipeline = MetaTunePipeline(self.test_csv, target_col='target')
        results = pipeline.run(epochs=1, train_brain=False)
        self.assertIsNotNone(results)
        self.assertIn('final_metric', results)

    def test_pipeline_cli_logic(self):
        # Mimic the stripped ARGPARSE logic
        with patch('sys.stdout', new=MagicMock()): # Suppress output
            test_args = ['pipeline.py', self.test_csv, '--target', 'target', '--epochs', '1']
            with patch.object(sys, 'argv', test_args):
                parser = argparse.ArgumentParser(description="MetaTune Pipeline")
                parser.add_argument("data", help="Path to CSV dataset")
                parser.add_argument("--target", help="Target column name", default=None)
                parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
                
                args = parser.parse_args()
                
                pipeline = MetaTunePipeline(args.data, target_col=args.target)
                results = pipeline.run(epochs=args.epochs)
                
                self.assertIsNotNone(results)

    @patch('engine.DynamicTrainer.run')
    def test_visualize_no_key_error(self, mock_run):
        pipeline = MetaTunePipeline(self.test_csv, target_col='target')
        pipeline.training_results = {
            "final_metric": 0.9, "metric_name": "accuracy", 
            "training_time": 1.0, "train_loss_history": [0.5, 0.4, 0.3],
            "val_loss_history": [0.6, 0.5, 0.45]
        }
        try:
            pipeline._visualize()
        except KeyError as e:
            self.fail(f"_visualize() raised KeyError: {e}")

if __name__ == '__main__':
    unittest.main()
