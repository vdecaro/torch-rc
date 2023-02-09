from torch.utils.data import ConcatDataset

from typing import List, Union

DATA_PATH = "/raid/decaro/datasets"


def get_dataset(name: str, users: Union[str, List[str]], continual: bool = False):

    if name == "WESAD":
        from .wesad import WESADDataset

        data_class, n_contexts = WESADDataset, 5

    elif name == "HHAR":
        from .hhar import HHARDataset

        data_class, n_contexts = HHARDataset, 4

    else:
        raise ValueError(f"Dataset {name} not found.")

    if isinstance(users, str):
        if not continual:
            data = ConcatDataset(
                [data_class(user=users, context=c) for c in range(n_contexts)]
            )
        else:
            data = [data_class(user=users, context=c) for c in range(n_contexts)]
    else:
        if not continual:
            data = ConcatDataset(
                [
                    data_class(user=u, context=c)
                    for u in users
                    for c in range(n_contexts)
                ]
            )
        else:
            data = [
                ConcatDataset([data_class(user=u, context=c) for u in users])
                for c in range(n_contexts)
            ]

    return data
