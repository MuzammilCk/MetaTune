
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from data_analyzer import DatasetAnalyzer
    from brain import MetaLearner 
    from engine import DynamicTrainer
except ImportError as e:
    print(f"❌ Error: Missing component files. {e}"); sys.exit(1)

class MetaTunePipeline:
    def __init__(self, data_path, target_col=None):
        self.data_path = data_path; self.target_col = target_col
        self.dataset_dna = None; self.predicted_params = None; self.training_results = None
        self.meta_learner = MetaLearner()
        
    def run(self, train_brain=False, epochs=20): # Defaults to 20 for speed
        print("\n" + "="*70 + "\n🚀 METATUNE ENTERPRISE PIPELINE STARTING\n" + "="*70)
        
        try:
            import time
            from vizier_stub import Study, Trial
            self.study = Study(name=f"metatune_run_{int(time.time())}")
        except ImportError:
            self.study = None
            
        # PHASE 1: DIAGNOSIS
        print("\n🔹 PHASE 1: Forensic Data Analysis")
        analyzer = DatasetAnalyzer(self.data_path, target_col=self.target_col)
        if not analyzer.load_data(): return None
        self.dataset_dna = analyzer.analyze()
        print(f"   ✓ Task Type: {self.dataset_dna.get('task_type', 'Unknown')}")
        print(f"   ✓ Complexity Score: {self.dataset_dna.get('target_entropy', 0):.3f}")

        # PHASE 2: PRESCRIPTION
        print("\n🔹 PHASE 2: Neural Hyperparameter Prediction")
        # Try to train brain if enough data exists
        self.meta_learner.train(epochs=30) 
        
        self.predicted_params = self.meta_learner.predict(self.dataset_dna)
        print("\n   ✨ OPTIMIZED CONFIGURATION GENERATED:")
        for k, v in self.predicted_params.items(): print(f"   ► {k:20s}: {v}")

        # PHASE 3: EXECUTION
        print("\n🔹 PHASE 3: Training Deployment")
        try:
            from bilevel import BilevelOptimizer, BilevelConfig
            from engine import DynamicTrainer
            
            # Use X_train, y_train implicitly inside training logic mapped
            import pandas as pd
            from sklearn.model_selection import train_test_split
            df = pd.read_csv(self.data_path)
            # Impute dummy target logic mapping
            target_col = self.target_col if self.target_col else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            task_type = self.dataset_dna.get('task_type', 'classification')

            config = BilevelConfig(max_outer_iterations=5, population_size=3)
            optimizer = BilevelOptimizer(meta_learner=self.meta_learner, config=config)
            
            best_params = optimizer.optimize(
                self.dataset_dna, X_train, y_train, X_val, y_val, task_type
            )
            trainer = DynamicTrainer(self.data_path, self.dataset_dna, best_params, target_col=self.target_col)
            self.training_results = trainer.run(epochs=epochs)
            
            if self.study is not None:
                try:
                    from vizier_stub import Trial
                    trial = Trial(id=len(self.study.get_trials()), parameters=best_params)
                    trial.complete(
                        metric_value=self.training_results.get('final_metric', 0.0),
                        elapsed_secs=self.training_results.get('training_time', 0.0)
                    )
                    self.study.add_trial(trial)
                except Exception as e:
                    print(f"⚠️  Trial tracking failed: {e}")
                    
        except ImportError as e:
            trainer = DynamicTrainer(self.data_path, self.dataset_dna, self.predicted_params, target_col=self.target_col)
            self.training_results = trainer.run(epochs=epochs)
        
        # PHASE 4: FEEDBACK LOOP (Online Learning)
        print("\n🔹 PHASE 4: Cognitive Feedback Loop")
        final_metric = self.training_results['final_metric']
        print(f"   ✓ Run Performance ({self.training_results['metric_name']}): {final_metric:.4f}")
        
        self.meta_learner.store_experience(self.dataset_dna, self.predicted_params, final_metric)
        
        self._generate_report(); self._visualize()
        return self.training_results

    def _generate_report(self):
        report = {"dna": {k: v.item() if hasattr(v, 'item') else v for k,v in self.dataset_dna.items()}, "params": self.predicted_params, "metrics": {"final_accuracy": float(self.training_results['final_metric']), "training_time": float(self.training_results['training_time'])}}
        
        if self.study is not None:
            report["vizier_trials"] = [
                {"params": t.parameters, "metric": t.final_measurement}
                for t in self.study.get_trials()
            ]
            
        with open("metatune_report.json", "w") as f: json.dump(report, f, indent=4)
        print("\n📄 Report generated: 'metatune_report.json'")

    def _visualize(self):
        try:
            train_hist = self.training_results.get('train_loss_history', [])
            val_hist = self.training_results.get('val_loss_history', [])
            
            if not train_hist:
                print("⚠️  No loss history available to visualize.")
                return

            plt.figure(figsize=(10, 5))
            plt.plot(train_hist, label='Train Loss', color='blue')
            plt.plot(val_hist, label='Val Loss', color='orange')
            plt.title(f"MetaTune Optimization Trajectory"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig("metatune_graph.png"); print("📊 Visualization saved: 'metatune_graph.png'")
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaTune Pipeline")
    parser.add_argument("data", help="Path to CSV dataset")
    parser.add_argument("--target", help="Target column name", default=None)
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    
    args = parser.parse_args()
    
    pipeline = MetaTunePipeline(args.data, target_col=args.target)
    pipeline.run(epochs=args.epochs)


