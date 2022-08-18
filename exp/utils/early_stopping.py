from collections import defaultdict
from typing import Dict
from ray.tune.stopper import Stopper

class TrialNoImprovementStopper(Stopper):

    def __init__(self,
                 metric: str,
                 mode: str = None,
                 patience_threshold: int = 10):
        self._metric = metric
        self._mode = mode
        self._patience_threshold = patience_threshold

        self._trial_patience = defaultdict(lambda: 0)
        if mode == 'min':
            self._trial_best = defaultdict(lambda: float('inf')) 
        else:
            self._trial_best = defaultdict(lambda: -float('inf'))

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)

        better = (self._mode == 'min' and metric_result < self._trial_best[trial_id]) or \
            (self._mode == 'max' and metric_result > self._trial_best[trial_id])

        if better:
            self._trial_best[trial_id] = metric_result
            self._trial_patience[trial_id] = 0
        else:
            self._trial_patience[trial_id] += 1
        
        return self._trial_patience[trial_id] >= self._patience_threshold
    
    def stop_all(self):
        return False
