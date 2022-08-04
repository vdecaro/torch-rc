from typing import List

import torch



class CentralizedDataset(torch.utils.data.Dataset):

    def __init__(self, name: str, idx: List[int]) -> None:
        super().__init__()
        self.users = idx
        if name == 'WESAD':
            from .wesad import WESADDataset
            self.data: List[WESADDataset] = [WESADDataset(i) for i in idx]
        if name == 'HHAR':
            from .hhar import HHARDataset
            self.data: List[HHARDataset] = [HHARDataset(i) for i in idx]

        self._seq_length = None

    @property
    def seq_length(self):
        return self._seq_length

    @seq_length.setter
    def seq_length(self, new_length: int):
        if self._seq_length is None or new_length != self._seq_length:
            for d in self.data:
                d.seq_length = new_length
            self._seq_length = new_length
    
    def __len__(self):
        return sum([len(d) for d in self.data])

    def __getitem__(self, i: int):
        data_idx = 0
        while i > len(self.data[data_idx]):
            i -= len(self.data[data_idx])
            data_idx += 1
        
        return self.data[data_idx][i]
    