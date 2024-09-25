import os
from ray import tune


def search_space():
    return {
        "root": os.path.join(os.getcwd(), "data"),
        "permute_seed": tune.grid_search([None, 42]),
        "input_size": 1,
        "out_size": 10,
        "block_sizes": [32 for _ in range(16)],
        "block_config": tune.grid_search([1, 2, 3, 4]),
        "coupling_block_init_fn": ("orthogonal",),
        "coupling_perc": tune.grid_search([20, 0.3]),
        "generalized_coupling": False,
        "eul_step": 0.03,
        "activation": "tanh",
        "decay_epochs": [80, 150],
        "decay_scalar": 0.1,
    }
