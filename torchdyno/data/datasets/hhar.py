import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import zscore

PHONES = ["nexus4", "s3", "s3mini"]  # , 'samsungold']
WATCHES = ["gear", "lgwatch"]
LABELS = {"stand": 0, "sit": 1, "walk": 2, "stairsup": 3, "stairsdown": 4, "bike": 5}
FREQ = {
    "nexus4": 200,
    "s3": 150,
    "s3mini": 100,
    "gear": 100,
    "lgwatch": 200,
}  #'samsungold': 50
TARGET_FREQ = 100
TOLERANCE = 5


class HHARDataset(torch.utils.data.Dataset):
    """HHAR dataset.

    The HHAR dataset is a dataset of human activity recognition (HAR) collected from
    smartphones and smartwatches. The dataset contains accelerometer and gyroscope data
    from different devices and users. The dataset is used to classify the activity of
    the user based on the sensor data.
    """

    USERS = {
        "all": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        "train": {
            25: ["a", "b"],
            50: ["a", "b", "c"],
            75: ["a", "b", "c", "e"],
            100: ["a", "b", "c", "e", "h"],
        },
        "eval": ["f", "i"],
        "test": ["d", "g"],
    }
    CONTEXTS = ["nexus4", "s3", "s3mini", "lgwatch"]

    def __init__(
        self, root: str, user: str, context: int, seq_length: Optional[int] = 200
    ):
        """Initialize the HHAR dataset.

        Args:
            root (str): Root directory of the dataset.
            user (str): User to load.
            context (int): Context to load.
            seq_length (Optional[int], optional): Sequence length. If None, the dataset
                is not split into sequences. Defaults to 200.
        """
        super().__init__()
        if user not in HHARDataset.USERS["all"]:
            raise ValueError(f"User {user} not found in HHAR dataset")
        if context < 0 or context >= len(HHARDataset.CONTEXTS):
            raise ValueError(f"Context {context} not found in HHAR dataset")

        self.root = root
        self.user = user
        self.context = HHARDataset.CONTEXTS[context]
        self.freq = TARGET_FREQ
        self._seq_length = seq_length

        self.path = os.path.join(self.processed_path, self.user, f"{self.context}.pkl")

        if not os.path.exists(self.path):
            os.makedirs(os.path.join(self.processed_path, self.user), exist_ok=True)
            self.preprocess()

        self.data = pickle.load(open(self.path, "rb"))

        self.features, self.targets = self.data["X"], self.data["Y"]
        if seq_length is not None:
            self.features, self.targets = self._to_sequence_chunks(seq_length)

    def __len__(self):
        """Return the length of the dataset."""
        if len(self.features.shape) == 2:
            return 1
        return self.features.shape[1]

    def __getitem__(self, i: int):
        """Return the item at the given index."""
        if len(self.features.shape) == 2:
            return self.features, self.targets
        return self.features[:, i], self.targets[:, i]

    def _to_sequence_chunks(self, length: int):
        X, Y = self.data["X"], self.data["Y"]
        X, Y = torch.split(X, length, dim=0), torch.split(Y, length, dim=0)
        if X[-1].shape[0] != length:
            X, Y = X[:-1], Y[:-1]
        return torch.stack(X, dim=1), torch.stack(Y, dim=1)

    def preprocess(self):
        print(f"Preprocessing {self.user} - {self.context} - {self.freq}Hz...")
        if self.context in PHONES:
            acc_path, gyr_path = "Phones_accelerometer.csv", "Phones_gyroscope.csv"
        else:
            acc_path, gyr_path = "Watch_accelerometer.csv", "Watch_gyroscope.csv"
        rdfs = {
            "acc": pd.read_csv(os.path.join(self.raw_path, acc_path)).dropna(
                subset=["gt"]
            ),
            "gyr": pd.read_csv(os.path.join(self.raw_path, gyr_path)).dropna(
                subset=["gt"]
            ),
        }
        dev = {"1": None, "2": None}
        for i in ["1", "2"]:
            dev[i] = self._merge_user_device(
                rdfs["acc"], rdfs["gyr"], self.user, f"{self.context}_{i}", TOLERANCE
            )

        dev_1_perc, dev_2_perc = dev["1"].isna().sum().max() / len(dev["1"]), dev[
            "2"
        ].isna().sum().max() / len(dev["2"])
        chosen_dev, perc = (
            ("1", dev_1_perc) if (dev_1_perc <= dev_2_perc) else ("2", dev_2_perc)
        )
        df_dev = dev[chosen_dev]
        if self.freq != FREQ[self.context]:
            down_idx = np.around(
                np.arange(0, len(df_dev) - 1, FREQ[self.context] / self.freq)
            )
            df_dev = df_dev.iloc[down_idx]
        df_dev = df_dev.dropna()

        values = ["x_acc", "y_acc", "z_acc", "x_gyr", "y_gyr", "z_gyr"]

        to_save = {
            "X": torch.tensor(df_dev[values].apply(zscore).values),
            "Y": F.one_hot(
                torch.tensor(df_dev["gt"].apply(lambda x: LABELS[x]).values),
                num_classes=6,
            ),
        }

        pickle.dump(to_save, open(self.path, "wb+"))
        print(f"{self.user} - {self.context} - {self.freq}Hz preprocessed!")

    def _merge_user_device(self, acc_df, gyr_df, user, device, tolerance):
        def aux(df, user, device, sd):
            return (
                df.loc[
                    ((df["User"] == user) & (df["Device"] == device)),
                    ["Arrival_Time", "x", "y", "z", "gt"],
                ]
                .sort_values("Arrival_Time")
                .rename(columns={"x": f"x_{sd}", "y": f"y_{sd}", "z": f"z_{sd}"})
            )

        ud_acc, ud_gyr = aux(acc_df, user, device, f"acc"), aux(
            gyr_df, user, device, f"gyr"
        )
        no_gt = ["Arrival_Time", "x_gyr", "y_gyr", "z_gyr"]
        lu_df = pd.merge_asof(
            ud_acc,
            ud_gyr[no_gt],
            on="Arrival_Time",
            direction="nearest",
            tolerance=tolerance,
        )
        no_gt = ["Arrival_Time", "x_acc", "y_acc", "z_acc"]
        ru_df = pd.merge_asof(
            ud_gyr,
            ud_acc[no_gt],
            on="Arrival_Time",
            direction="nearest",
            tolerance=tolerance,
        )
        u_df = pd.concat([lu_df, ru_df[ru_df["x_acc"].isnull()]]).sort_values(
            "Arrival_Time"
        )
        u_df = u_df[
            ["Arrival_Time", "x_acc", "y_acc", "z_acc", "x_gyr", "y_gyr", "z_gyr", "gt"]
        ]
        return u_df

    @property
    def seq_length(self):
        """Return the sequence length."""
        return self._seq_length

    @seq_length.setter
    def seq_length(self, new_length: int):
        """Set the sequence length.

        If the sequence length is different from the current one, the dataset is split
        into sequences of the new length.
        """
        if self._seq_length is None or new_length != self._seq_length:
            self.features, self.targets = self._to_sequence_chunks(new_length)
            self._seq_length = new_length

    @property
    def raw_path(self):
        return os.path.join(self.root, "raw", "HHAR")

    @property
    def processed_path(self):
        return os.path.join(self.root, "processed", "HHAR")
