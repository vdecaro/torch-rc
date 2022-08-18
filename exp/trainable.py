import os
from ray import tune
import torch

from typing import Dict, Union

from torch_esn.model.reservoir import Reservoir
from torch_esn.optimization.intrinsic_plasticity import IntrinsicPlasticity
from torch_esn.optimization.norm_ridge_regression import fit_and_validate_readout

from .data.centralized_dataset import CentralizedDataset
from .data.seq_loader import seq_collate_fn
from torch.utils.data import DataLoader


class ESNTrainable(tune.Trainable):

    def setup(self, config: Dict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mode = config['MODE']
        self.l2_values = None

        self.train_data = CentralizedDataset(config['dataset'], idx=config['TRAIN_USERS'])
        self.train_data.seq_length = config['SEQ_LENGTH']
        self.eval_data = CentralizedDataset(config['dataset'], idx=config['VALIDATION_USERS'])
        self.eval_data.seq_length = config['SEQ_LENGTH']
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=config['BATCH_SIZE'],
            shuffle=True,
            collate_fn=seq_collate_fn
        )
        self.eval_loader = DataLoader(
            self.eval_data,
            batch_size=500,
            shuffle=False,
            collate_fn=seq_collate_fn
        )
        
        self.reservoir = Reservoir(
            input_size = config['INPUT_SIZE'],
            hidden_size=config['HIDDEN_SIZE'],
            activation='tanh',
            leakage=config['LEAKAGE'],
            input_scaling=config['INPUT_SCALING'],
            rho=config['RHO'],
            mu=config['MU'],
            sigma=config['SIGMA'],
            mode=config['MODE']
        )
        self.readout: torch.Tensor = None

        if self.mode == 'intrinsic_plasticity':
            self.ip_opt: IntrinsicPlasticity = IntrinsicPlasticity(
                config['ETA'], 
                config['MU'], 
                config['SIGMA']
            )
    
    def step(self):
        if self.mode == 'intrinsic_plasticity':
            self.run_ip_epoch()
        metrics = self.run_readout_epoch()
        return metrics
    
    @torch.no_grad
    def run_ip_epoch(self):
        self.reservoir = self.reservoir.train()
        for x, _ in self.train_loader:
            self.reservoir(x.to(self.device))
            self.ip_opt.backward()
            self.ip_opt.step()
            self.reservoir.zero_grad()

    @torch.no_grad
    def run_readout_epoch(self):
        self.reservoir = self.reservoir.eval()
        self.readout, acc, _ = fit_and_validate_readout(
            train_loader=self.train_loader,
            eval_loader=self.eval_loader,
            l2_values=self.get_config['l2'],
            score_fn=acc_fn,
            mode='max',
            preprocess_fn=self.reservoir.forward
        )
        return {
            'eval_score': acc
        }

    def save_checkpoint(self, tmp_checkpoint_dir: str):
        torch.save(self.reservoir, os.path.join(tmp_checkpoint_dir, 'reservoir.pkl'))
        torch.save(self.readout, os.path.join(tmp_checkpoint_dir, 'readout.pkl'))
        return tmp_checkpoint_dir
        
    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        self.reservoir = torch.load(os.path.join(checkpoint, 'reservoir.pkl'))
        self.readout = torch.load(os.path.join(checkpoint, 'readout.pkl'))
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        torch.save(self.reservoir, os.path.join(tmp_checkpoint_dir, 'reservoir.pkl'))
        torch.save(self.readout, os.path.join(tmp_checkpoint_dir, 'readout.pkl'))
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, checkpoint):
        self.reservoir = torch.load(os.path.join(checkpoint, 'reservoir.pkl'))
        self.readout = torch.load(os.path.join(checkpoint, 'readout.pkl'))
    

def acc_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.argmax(-1).flatten().cpu()
    y_pred = y_pred.argmax(-1).flatten().cpu()
    return (torch.sum(y_true == y_pred)/y_true.size(0)).item()
