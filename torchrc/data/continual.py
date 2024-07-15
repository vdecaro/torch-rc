import torch
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.benchmarks.utils import AvalancheDataset
from .datasets import get_dataset

from typing import List, Union


def continual_benchmark(name: str, users: Union[str, List[str]]):
    def _collate_fn(batch):
        x, y = [], []
        for x_i, y_i, _ in batch:
            x.append(x_i)
            y.append(y_i)

        return torch.stack(x, dim=1), torch.stack(y, dim=1)

    data = get_dataset(name, users, continual=True)
    train_exps = []
    for i, data_i in enumerate(data):
        tl = DataAttribute(
            ConstantSequence(i, len(data_i)), "targets_task_labels", use_in_getitem=True
        )
        exp = CLExperience()
        exp.dataset = AvalancheDataset(
            [data_i], data_attributes=[tl], collate_fn=_collate_fn
        )
        train_exps.append(exp)

    return CLScenario([CLStream("full", train_exps)])
