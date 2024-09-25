import os
from ray import tune
from ray import train
from ray.tune import stopper

from ray.tune import CLIReporter

# from invalid_stopper import InvalidStopper
from search_space import search_space
from trainable import RNNofRNNsTrainable


def main():
    epoch_stopper = stopper.MaximumIterationStopper(200)
    trial_plateau = stopper.TrialPlateauStopper(
        metric="test_acc",
        std=0.001,
        num_results=5,
        grace_period=50,
    )
    # invalid_stopper = InvalidStopper("test_acc", -1.0)
    stoppers = stopper.CombinedStopper(
        epoch_stopper, trial_plateau
    )  # , invalid_stopper)

    reporter = CLIReporter(
        parameter_columns={
            "permute_seed": "seed",
            "generalized_coupling": "gen_coupling",
            "activation": "act",
            "adapt_blocks": "adapt",
            "squash_blocks": "squash",
            "orthogonal_blocks": "orthogonal",
        },
        metric_columns=[
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
        ],
    )
    run_config = train.RunConfig(
        storage_path=os.path.join(os.getcwd(), "results"),
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="test_acc",
            checkpoint_score_order="max",
            checkpoint_frequency=1,
        ),
        stop=stoppers,
        progress_reporter=reporter,
    )

    tune_config = tune.TuneConfig(num_samples=3)
    tuner = tune.Tuner(
        tune.with_resources(RNNofRNNsTrainable, {"cpu": 1, "gpu": 0.5}),
        param_space=search_space(),
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()


if __name__ == "__main__":
    main()
