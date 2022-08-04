import os
import pickle
import pathlib

import torch
import torch.nn.functional as F
import numpy as np

RAW_WESAD_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'raw', 'WESAD')
WESAD_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'processed', 'WESAD')

USERS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14", "15", "16", "17"]

class WESADDataset(torch.utils.data.Dataset):

    def __init__(self, idx: int) -> None:
        super().__init__()
        self.user = USERS[idx]

        u_path = os.path.join(WESAD_PATH, f'{self.user}.pkl')
        if not os.path.exists(u_path):
            self.user_data = self.preprocess()
        else:
            self.user_data = pickle.load(open(u_path, 'rb'))

        self._seq_length = None
        self.X, self.Y = None, None

    @property
    def seq_length(self):
        return self._seq_length

    @seq_length.setter
    def seq_length(self, new_length: int):
        if self._seq_length is None or new_length != self._seq_length:
            print(f"Setting the length of the chunks in WESAD user {self.user} from {self._seq_length} to {new_length}")
            self.X, self.Y = self._to_sequence_chunks(new_length)
            self._seq_length = new_length

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, i: int):
        return self.X[:, i], self.Y[:, i]
    
    def _to_sequence_chunks(self, length: int):
        X, Y = torch.split(self.user_data['X'], length, dim=0), torch.split(self.user_data['Y'], length, dim=0)
        if X[-1].shape[0] != length:
            X, Y = X[:-1], Y[:-1]

        return torch.stack(X, dim=1), torch.stack(Y, dim=1)

    def preprocess(self):
        print(f"Preprocessing user {self.user}...")
        with open(os.path.join(RAW_WESAD_PATH, f'S{self.user}', f'S{self.user}.pkl'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        X = np.concatenate([
            data['signal']['chest']['ACC'],
            data['signal']['chest']['Resp'],
            data['signal']['chest']['EDA'],
            data['signal']['chest']['ECG'],
            data['signal']['chest']['EMG'],
            data['signal']['chest']['Temp'],
        ], axis=1)
        Y = data['label']
        X = X[(Y>0) & (Y<5)]
        X = torch.tensor((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        Y = Y[(Y>0) & (Y<5)]
        Y = F.one_hot(torch.tensor(Y-1, dtype=torch.int64), num_classes=4)
        u_dict = {'X': X, 'Y': Y}
        print(f"Preprocessing of user {self.user} complete!")

        print(f"Saving preprocessed data of user {self.user}...")
        pickle.dump(u_dict, open(os.path.join(WESAD_PATH, f'{self.user}.pkl'), 'wb+'))
        print(f"Preprocessed data of user {self.user} saved!")

        return u_dict
