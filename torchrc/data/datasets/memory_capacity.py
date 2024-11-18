import torch
from torch import nn
from torch.utils.data import Dataset


class MemoryCapacityDataset(Dataset):

    TRAIN_SIZE = 5000
    TEST_SIZE = 1000

    def __init__(
        self,
        delay: int,
        length: int = 6000,
        seed: int = 0,
        return_full_sequence: bool = False,
    ):
        """Memory capacity dataset.

        Args:
            delay: The delay between the input and the target. Ideally, 2*hidden_size of
                the evaluated RNN.
            length: The length of the dataset.
            seed: Random seed.
        """
        self.length = length
        self.delay = delay
        self.seed = seed
        self.return_full_sequence = return_full_sequence

        self.data = self._generate_data()

    def __len__(self):
        if self.return_full_sequence:
            return 1
        return self.length - self.delay

    def __getitem__(self, idx: int):
        if self.return_full_sequence:
            target = []
            for i in range(self.delay + 1):
                target.append(self.data[:-i])

            return self.data[self.delay :], torch.stack(
                [self.data[:-i] for i in range(self.delay + 1)]
            )
        return self.data[idx + self.delay], self.data[idx]

    def _generate_data(self):
        torch.manual_seed(self.seed)
        data = torch.empty(self.length).uniform_(-0.8, 0.8)
        return data
