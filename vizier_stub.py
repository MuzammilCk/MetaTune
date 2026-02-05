
import uuid
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Union

class TrialState(Enum):
    REQUESTED = "REQUESTED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    STOPPING = "STOPPING"

class Parameter:
    def __init__(self, value: Union[int, float, str]):
        self.value = value

class Measurement:
    def __init__(self, metrics: Dict[str, float] = None, step: int = 0):
        self.metrics = metrics or {}
        self.step = step

class Trial:
    def __init__(self, parameters: Dict[str, Any], trial_id: int):
        self.id = trial_id
        self.parameters = parameters
        self.state = TrialState.REQUESTED
        self.final_measurement: Optional[Measurement] = None
        self.measurements: List[Measurement] = []
        self.created_time = time.time()
        self.completed_time = None

    def complete(self, measurement: Measurement):
        self.final_measurement = measurement
        self.state = TrialState.COMPLETED
        self.completed_time = time.time()
        
    def should_stop(self) -> bool:
        return self.state == TrialState.STOPPING

class Designer:
    """Abstract base class for algorithms (Designers)."""
    def suggest(self, count: int = 1) -> List[Trial]:
        raise NotImplementedError
    
    def update(self, completed: Trial, all_trials: List[Trial]):
        pass

class Study:
    """Manages the optimization run."""
    def __init__(self, designer: Designer, study_id: str = "default_study"):
        self.study_id = study_id
        self.designer = designer
        self.trials: List[Trial] = []
        self._trial_counter = 0

    def suggest(self, count: int = 1) -> List[Trial]:
        """Asks the designer for new suggestions."""
        new_trials = self.designer.suggest(count)
        for t in new_trials:
            self._trial_counter += 1
            t.id = self._trial_counter
            self.trials.append(t)
        return new_trials

    def trials(self) -> List[Trial]:
        return self.trials

    def optimal_trials(self) -> List[Trial]:
        # Simple heuristic: filter for completed and sort by 'final_metric' or 'accuracy'
        # Assuming higher is better for now, or check metric definition
        completed = [t for t in self.trials if t.state == TrialState.COMPLETED]
        if not completed: return []
        # Sort by the first metric found
        return sorted(completed, key=lambda t: list(t.final_measurement.metrics.values())[0], reverse=True)
