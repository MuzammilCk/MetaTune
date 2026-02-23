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
    def __init__(self, metrics: Dict[str, float] = None, step: int = 0, elapsed_secs: float = 0.0):
        self.metrics = metrics or {}
        self.step = step
        self.elapsed_secs = elapsed_secs

class Trial:
    def __init__(self, parameters: Dict[str, Any], trial_id: int = None, id: int = None):
        self.id = trial_id if trial_id is not None else id
        self.parameters = parameters
        self.state = TrialState.REQUESTED
        self.final_measurement: float = None
        self.measurements: List[Measurement] = []
        self.created_time = time.time()
        self.completed_time = None
        self.elapsed_secs: float = 0.0

    @property
    def status(self) -> str:
        return "COMPLETED" if self.final_measurement is not None else "ACTIVE"

    def complete(self, metric_value: float, elapsed_secs: float = 0.0) -> None:
        self.final_measurement = metric_value
        self.elapsed_secs = elapsed_secs
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

class SearchSpace:
    def __init__(self):
        self.parameters = {}
        
    def validate(self, params: dict) -> bool:
        """Return True if all provided params fall within defined bounds."""
        for name, config in self.parameters.items():
            if name in params:
                val = params[name]
                if hasattr(config, 'bounds') and config.bounds:
                    lo, hi = config.bounds[0], config.bounds[1]
                    if not (lo <= val <= hi):
                        return False
        return True

class Study:
    """Manages the optimization run."""
    def __init__(self, designer: Designer = None, study_id: str = "default_study", name: str = None):
        self.study_id = study_id if name is None else name
        self.designer = designer
        self._trials: List[Trial] = []
        self._trial_counter = 0

    def suggest(self, count: int = 1) -> List[Trial]:
        """Asks the designer for new suggestions."""
        if self.designer is None:
            return []
        new_trials = self.designer.suggest(count)
        for t in new_trials:
            self._trial_counter += 1
            t.id = self._trial_counter
            self._trials.append(t)
        return new_trials

    def get_trials(self) -> List[Trial]:
        return self._trials

    @property
    def trials(self):
        """Backward-compatible read access to trial list."""
        return self._trials

    def add_trial(self, trial) -> None:
        self._trials.append(trial)

    def optimal_trials(self) -> list:
        completed = [t for t in self._trials if t.final_measurement is not None]
        if not completed:
            return []
        return sorted(completed, key=lambda t: t.final_measurement, reverse=True)[:1]
