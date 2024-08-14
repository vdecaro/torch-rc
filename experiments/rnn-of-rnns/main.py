import os
from ray import tune
from ray import train
from ray.tune import stopper

from .invalid_stopper import InvalidStopper
from .search_space import search_space
from .trainable import RNNofRNNsTrainable


def main():
    epoch_stopper = stopper.MaximumIterationStopper(200)
    trial_plateau = stopper.TrialPlateauStopper(
        metric="test_loss",
        float=0.01,
        num_results=10,
        grace_period=50,
        mode="min",
    )
    invalid_stopper = InvalidStopper("test_acc", -1.0)
    stoppers = stopper.CombinedStopper(epoch_stopper, trial_plateau, invalid_stopper)
    run_config = train.RunConfig(
        storage_path=os.path.join(os.getcwd(), "experiments"),
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="test_acc",
            checkpoint_score_order="max",
            checkpoint_frequency=1,
        ),
        stop=stoppers,
    )

    tune_config = tune.TuneConfig(
        num_samples=5,
        reuse_actors=True,
    )
    tuner = tune.Tuner(
        tune.with_resources(RNNofRNNsTrainable, {"cpu": 1, "gpu": 0.3}),
        param_space=search_space(),
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


if __name__ == "__main__":
    main()
