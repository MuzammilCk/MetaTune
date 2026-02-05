
from vizier_stub import Designer, Trial
from brain import MetaLearner
import uuid

class MetaLearningDesigner(Designer):
    """
    Wraps the MetaLearner 'Brain' into a Vizier Designer.
    Adapts the predict() method to return Trails.
    """
    def __init__(self, dataset_dna):
        self.dna = dataset_dna
        self.brain = MetaLearner()
        # Try load existing brain
        try:
             self.brain = MetaLearner.load("meta_brain.pkl")
        except:
             pass # Use untrained/heuristics if load fails
        
        if not self.brain.is_trained:
             # Quick train if needed, or rely on heuristics
             pass

    def suggest(self, count: int = 1):
        trials = []
        for _ in range(count):
            # 1. Get Params from Brain
            # Note: predict() includes some exploration noise usually
            params = self.brain.predict(self.dna)
            
            # 2. Create Trial
            # We don't set ID here, Study sets it
            t = Trial(parameters=params, trial_id=0) 
            trials.append(t)
        return trials

    def update(self, completed_trial: Trial, all_trials):
        """
        Feedback Loop: Update the Brain with the results.
        """
        if completed_trial.state != "COMPLETED" or not completed_trial.final_measurement:
            return

        # Extract metric
        # Assuming the first metric is the objective
        metrics = completed_trial.final_measurement.metrics
        main_metric = list(metrics.values())[0] if metrics else 0.0
        
        t_params = completed_trial.parameters
        
        # Store in Brain's Knowledge Base
        self.brain.store_experience(self.dna, t_params, main_metric)
        
        # Online Learning? 
        # Ideally we retrain the brain here, but for now we just store.
