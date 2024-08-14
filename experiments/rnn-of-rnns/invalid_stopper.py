from ray.tune.stopper import Stopper


class InvalidStopper(Stopper):
    def __init__(self, metric: str, invalid_value: float):
        self.metric = metric
        self.invalid_value = invalid_value

    def __call__(self, trial_id, result):
        return result[self.metric] == self.invalid_value
