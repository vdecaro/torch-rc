import torch
from torch.utils.data import DataLoader

from torchrc.data.utils.seq_loader import seq_collate_fn
from torchrc.data.datasets.seq_mnist import SequentialMNIST


def load_mnist(train_batch_size: int, test_batch_size: int):
    train_data = SequentialMNIST("./data", train=True, download=True)
    test_data = SequentialMNIST("./data", train=False, download=True)

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=seq_collate_fn(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=seq_collate_fn(),
    )
    return train_loader, test_loader
