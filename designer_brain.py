
from vizier_stub import Designer, Trial, TrialState
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
             self.brain = MetaLearner.load("meta_brain_weights.pth")
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
        if completed_trial.state != TrialState.COMPLETED or not completed_trial.final_measurement:
            return
        main_metric = float(completed_trial.final_measurement)
        self.brain.store_experience(self.dna, completed_trial.parameters, main_metric)
