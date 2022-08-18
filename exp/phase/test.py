import pickle
import os
from typing import Literal

import ray
from ray import tune
from ..data.seq_loader import seq_collate_fn

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..data.centralized_dataset import CentralizedDataset
import exp.data.split_config as cfg


@torch.no_grad()
def run(dataset: str,
        perc: int,
        mode: Literal['vanilla', 'intrinsic_plasticity']):

    retrain_dir = f"experiments/{dataset}_{perc}_{mode}/retraining"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    retrain_exp = tune.ExperimentAnalysis(retrain_dir, default_metric='eval_score', default_mode='max')
    config = retrain_exp.get_best_config()

    test_data = CentralizedDataset(dataset, cfg.USERS[dataset]['TEST'])
    test_data.seq_length = config['SEQ_LENGTH']
    test_loader = DataLoader(
            test_data, 
            batch_size=500,
            collate_fn=seq_collate_fn
    )
    acc_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()

    acc = {}
    for i, trial in enumerate(retrain_exp.trials):
        chk: ray.ml.Checkopoint = retrain_exp.get_best_checkpoint(trial)
        reservoir = torch.load(os.path.join(chk.local_path, 'reservoir.pkl')).to(device).eval()
        readout = torch.load(os.path.join(chk.local_path, 'readout.pkl')).to(device)
        trial_acc, trial_n_samples = 0, 0

        for x, y in test_loader:
            h = reservoir(x.to('cuda' if torch.cuda.is_available() else 'cpu')).reshape((-1, reservoir.hidden_size))
            Y_pred = torch.argmax(F.linear(h, readout), -1).flatten().to('cpu')
            Y_true = torch.argmax(y, dim=-1).flatten()
            curr_acc = acc_fn(Y_true, Y_pred)
            curr_n_samples = Y_true.size(0)
            trial_acc += curr_acc * curr_n_samples
            trial_n_samples += curr_n_samples
        acc[f'trial_{i}'] = trial_acc / trial_n_samples
    with open(f"experiments/{dataset}_{perc}_{mode}/test_res.pkl", 'wb+') as f:
        pickle.dump(acc, f)
    print(acc, "saved.")
