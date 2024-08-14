import os
from ray import tune


def search_space():
    return {
        "data_params": {
            "root": os.path.join(os.getcwd(), "data"),
            "permute_seed": tune.grid_search([None, 42]),
        },
        "rnn_params": {
            "input_size": 1,
            "out_size": 10,
            "block_sizes": [32 for _ in range(16)],
            "block_init_fn": tune.grid_search(
                [
                    ("sparse", 0.03),
                    ("sparse", 0.1),
                    ("orthogonal", 0.9),
                    ("diagonal",),
                ]
            ),
            "coupling_block_init_fn": ("orthogonal", 0.9),
            "coupling_perc": tune.grid_search([20, 0.1]),
            "generalized_coupling": tune.grid_search([False, True]),
            "eul_step": 0.03,
            "activation": tune.grid_search(["tanh", "relu", "selfnorm"]),
            "adapt_blocks": tune.grid_search([False, True]),
            "squash_blocks": tune.grid_search([None, "tanh", "clip"]),
            "orthogonal_blocks": tune.grid_search([False, True]),
        },
        "opt_params": {
            "decay_epochs": [80, 150],
            "decay_scalar": 0.1,
        },
    }
