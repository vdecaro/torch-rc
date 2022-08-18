import os
import pickle
import pathlib
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

RAW_HHAR_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets', 'raw', 'HHAR')
HHAR_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets', 'processed', 'HHAR')

USERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
PHONES = ['nexus4']#, 's3', 's3mini', 'samsungold']
WATCHES = []#['gear', 'lgwatch']
LABELS = {'stand': 0, 'sit': 1, 'walk': 2, 'stairsup': 3, 'stairsdown': 4, 'bike': 5}
FREQ = {'nexus4': 200, 's3': 150, 's3mini': 100, 'samsungold': 50, 'gear': 100, 'lgwatch': 200}
TARGET_FREQ = 50
TOLERANCE = 5


class HHARDataset(torch.utils.data.Dataset):

    def __init__(self, idx: int) -> None:
        super().__init__()
        self.user = USERS[idx]

        u_path = os.path.join(HHAR_PATH, f'{self.user}.pkl')
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
            print(f"Setting the length of the chunks in HHAR user {self.user} from {self._seq_length} to {new_length}")
            self.X, self.Y = self._to_sequence_chunks(new_length)
            self._seq_length = new_length
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]
    
    def _to_sequence_chunks(self, length: int):
        X, Y = [], []
        for k in self.user_data:
            dev_x, dev_y = self.user_data[k]['X'], self.user_data[k]['Y']
            dev_x, dev_y = torch.split(dev_x, length, dim=0), torch.split(dev_y, length, dim=0)
            if dev_x[-1].shape[0] != length:
                dev_x, dev_y = dev_x[:-1], dev_y[:-1]
            X += dev_x
            Y += dev_y
        return X, Y
    
    def preprocess(self):
        rdfs = {
            'p_acc': pd.read_csv(os.path.join(RAW_HHAR_PATH, 'Phones_accelerometer.csv')).dropna(subset=['gt']),
            'p_gyr': pd.read_csv(os.path.join(RAW_HHAR_PATH, 'Phones_gyroscope.csv')).dropna(subset=['gt']),
            #'w_acc': pd.read_csv(os.path.join(RAW_HHAR_PATH, 'Watch_accelerometer.csv')).dropna(subset=['gt']),
            #'w_gyr': pd.read_csv(os.path.join(RAW_HHAR_PATH, 'Watch_gyroscope.csv')).dropna(subset=['gt'])
        }
        u_dict = OrderedDict()
        curr_idx = 0
        for device in PHONES + WATCHES:
            dev = {'1': None, '2': None}
            for i in ['1', '2']:
                dev[i] = self._merge_user_device(
                    rdfs[f"{'p' if device in PHONES else 'w'}_acc"],
                    rdfs[f"{'p' if device in PHONES else 'w'}_gyr"],
                    self.user, 
                    f"{device}_{i}",
                    TOLERANCE
                )
            dev_1_perc, dev_2_perc = dev['1'].isna().sum().max() / len(dev['1']), dev['2'].isna().sum().max() / len(dev['2'])
            chosen_dev, perc = ('1', dev_1_perc) if (dev_1_perc <= dev_2_perc) else ('2', dev_2_perc)
            if perc < 0.1:
                df_dev = dev[chosen_dev]
                #down_idx = np.around(np.arange(0, len(df_dev)-1, FREQ[device]/TARGET_FREQ))
                #df_dev = df_dev.iloc[down_idx]
                df_dev = df_dev.dropna()
                values = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']
                df_dev[values] = (df_dev[values] - df_dev[values].mean()) / df_dev[values].std()
                df_dev['gt'] = df_dev['gt'].apply(lambda x: LABELS[x])
                
                u_dict[device] = {
                    'X': torch.tensor(df_dev[values].values),
                    'Y': F.one_hot(torch.tensor(df_dev['gt'].values), num_classes=6),
                    'first_index': curr_idx,
                    'size': len(df_dev)
                }
                curr_idx += len(df_dev)

        pickle.dump(u_dict, open(os.path.join(HHAR_PATH, f'{self.user}.pkl'), 'wb+'))
        
        return u_dict
    
    def _merge_user_device(self, acc_df, gyr_df, user, device, tolerance):
    
        def aux(df, user, device, sd):
            return df.loc[((df['User'] == user) & (df['Device'] == device)), ["Arrival_Time", "x", "y", "z", "gt"]] \
                .sort_values('Arrival_Time') \
                .rename(columns={"x": f"x_{sd}", "y": f"y_{sd}", "z": f"z_{sd}"})
        
        ud_acc, ud_gyr = aux(acc_df, user, device, f'acc'), aux(gyr_df, user, device, f'gyr')
        no_gt = ['Arrival_Time', 'x_gyr', 'y_gyr', 'z_gyr']
        lu_df = pd.merge_asof(ud_acc, ud_gyr[no_gt], on='Arrival_Time', direction='nearest', tolerance=tolerance)
        no_gt = ['Arrival_Time', 'x_acc', 'y_acc', 'z_acc']
        ru_df = pd.merge_asof(ud_gyr, ud_acc[no_gt], on='Arrival_Time', direction='nearest', tolerance=tolerance)
        u_df = pd.concat([lu_df, ru_df[ru_df['x_acc'].isnull()]]).sort_values('Arrival_Time')
        u_df = u_df[['Arrival_Time', 'x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'gt']]
        return u_df
    