import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class WESADDataset(torch.utils.data.Dataset):
    """WESAD dataset.

    The WESAD dataset is a dataset of physiological signals collected from a wristband
    and a chest strap. The dataset contains data from 15 subjects performing different
    activities. The dataset is used to classify the cognitive state of the user based on
    the physiological signals.
    """

    USERS = {
        "all": [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "13",
            "14",
            "15",
            "16",
            "17",
        ],
        "train": {
            25: ["2", "5", "7"],
            50: ["2", "5", "7", "8", "11"],
            75: ["2", "5", "7", "8", "11", "13", "15"],
            100: ["2", "5", "7", "8", "11", "13", "15", "16", "17"],
        },
        "eval": ["3", "9", "14"],
        "test": ["4", "6", "10"],
    }
    CONTEXTS = [0, 1, 2, 3, 4]

    def __init__(
        self, root: str, user: str, context: int, seq_length: Optional[int] = 700
    ) -> None:
        super().__init__()
        if user not in WESADDataset.USERS["all"]:
            raise ValueError(f"User {user} not found in WESAD dataset")
        if context not in WESADDataset.CONTEXTS:
            raise ValueError(f"Context {context} not found in WESAD dataset")
        self.root = root
        self.user = user
        self.context = context
        self._seq_length = seq_length

        self.path = os.path.join(self.processed_path, f"{self.user}.pkl")
        if not os.path.exists(self.path):
            self.preprocess()
        self.data = pickle.load(open(self.path, "rb"))
        self._get_context(context)

        self.features, self.targets = self.data["X"], self.data["Y"]
        if seq_length is not None:
            self.features, self.targets = self._to_sequence_chunks(seq_length)

    def __len__(self):
        if len(self.features.shape) == 2:
            return 1
        return self.features.shape[1]

    def __getitem__(self, i: int):
        if len(self.features.shape) == 2:
            return self.features, self.targets
        return self.features[:, i], self.targets[:, i]

    def preprocess(self):
        print(f"Preprocessing user {self.user}...")
        with open(
            os.path.join(self.raw_path, f"S{self.user}", f"S{self.user}.pkl"), "rb"
        ) as f:
            data = pickle.load(f, encoding="latin1")
        X = np.concatenate(
            [
                data["signal"]["chest"]["ACC"],
                data["signal"]["chest"]["Resp"],
                data["signal"]["chest"]["EDA"],
                data["signal"]["chest"]["ECG"],
                data["signal"]["chest"]["EMG"],
                data["signal"]["chest"]["Temp"],
            ],
            axis=1,
        )
        Y = data["label"]
        X = X[(Y > 0) & (Y < 5)]
        X = torch.tensor((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        Y = Y[(Y > 0) & (Y < 5)]
        Y = F.one_hot(torch.tensor(Y - 1, dtype=torch.int64), num_classes=4)
        u_dict = {"X": X, "Y": Y}
        print(f"Preprocessing of user {self.user} complete!")

        print(f"Saving preprocessed data of user {self.user}...")
        pickle.dump(u_dict, open(self.path, "wb+"))
        print(f"Preprocessed data of user {self.user} saved!")

    def _to_sequence_chunks(self, length: int):
        X, Y = self.data["X"], self.data["Y"]
        X, Y = torch.split(X, length, dim=0), torch.split(Y, length, dim=0)
        if X[-1].shape[0] != length:
            X, Y = X[:-1], Y[:-1]
        return torch.stack(X, dim=1), torch.stack(Y, dim=1)

    def _get_context(self, context: int):
        if context >= 5:
            raise ValueError("Context value must be < 5.")
        ctx_idx = np.where(np.diff(self.data["Y"].argmax(-1)) != 0)[0] + 1
        self.data["X"], self.data["Y"] = (
            np.split(self.data["X"], ctx_idx, axis=0)[context],
            np.split(self.data["Y"], ctx_idx, axis=0)[context],
        )

    @property
    def seq_length(self):
        return self._seq_length

    @seq_length.setter
    def seq_length(self, new_length: int):
        if self._seq_length is None or new_length != self._seq_length:
            self.features, self.targets = self._to_sequence_chunks(new_length)
            self._seq_length = new_length

    @property
    def raw_path(self):
        return os.path.join(self.root, "raw", "WESAD")

    @property
    def processed_path(self):
        return os.path.join(self.root, "processed", "WESAD")
