import numpy as np
import random
from copy import deepcopy

from .dataset import Digit5
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.scenarios import GenericCLScenario

from typing import List, Literal

__all__ = ["digit5_benchmark"]


RANDOM_SEED = 42


def digit5_benchmark(
    train_dataset: Digit5,
    test_dataset: Digit5,
    n_experiences: int,
    mode: Literal["exclusive", "uniform"] = "exclusive",
) -> GenericCLScenario:

    if mode not in ["exclusive", "uniform"]:
        raise ValueError("mode parameter must be in ['exclusive', 'uniform'].")

    train_exp_idx = _chunk_to_experience_indices(
        train_dataset.indices, mode, n_experiences
    )
    test_exp_idx = _chunk_to_experience_indices(
        test_dataset.indices, mode, n_experiences
    )

    train_sequence, test_sequence = [], []
    for i in range(n_experiences):
        train_data_exp, test_data_exp = deepcopy(train_dataset), deepcopy(test_dataset)
        train_data_exp.indices, test_data_exp.indices = (
            train_exp_idx[i],
            test_exp_idx[i],
        )
        train_sequence.append(train_data_exp)
        test_sequence.append(test_data_exp)

    return dataset_benchmark(train_datasets=train_sequence, test_datasets=test_sequence)


def _chunk_to_experience_indices(
    indices: List[List[int]], mode: str, n_experiences: int
):
    n_datasets = len(indices)
    if n_experiences % n_datasets != 0:
        raise ValueError(
            f"Parameter n_experiences must be a multiple of the number of datasets. \
                Found {n_experiences} and {n_datasets}"
        )

    # Depending on the distribution of the data across experiences (1 subdataset
    # per experience or uniform), we determine the number of chunks for each sub-dataset
    n_data_exp_chunks = (
        n_experiences // n_datasets if mode == "exclusive" else n_experiences
    )

    # We split all the sub-dataset in the desired chunks
    chunked_indices = []
    for d_idx in indices:
        chunked_d = [
            c.tolist() for c in np.array_split(np.array(d_idx), n_data_exp_chunks)
        ]
        # We shuffle the chunks of each sub-dataset locally
        random.Random(RANDOM_SEED).shuffle(chunked_d)
        chunked_indices.append(chunked_d)

    exp_indices = []
    for e in range(n_experiences):
        if mode == "exclusive":
            # We select one sub-dataset for each experience
            exp_idx = [[] for _ in range(n_datasets)]
            data_idx = e % n_datasets
            exp_idx[data_idx] = chunked_indices[data_idx].pop(0)
        elif mode == "uniform":
            # We get one chunk from each sub-dataset
            exp_idx = [
                chunked_indices[data_idx].pop(0) for data_idx in range(n_datasets)
            ]
        exp_indices.append(exp_idx)

    # Setting the random seed in the shuffle should ensure that, in the 'exclusive'
    # case, the training sequence and the test sequence will have the same order of
    # subdatasets
    random.Random(RANDOM_SEED).shuffle(exp_indices)

    return exp_indices
