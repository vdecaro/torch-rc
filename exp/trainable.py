from ray import tune
import torch

from torch_esn.model.reservoir import Reservoir
from torch_esn.optimization.intrinsic_plasticity import IntrinsicPlasticity
from torch_esn.optimization.norm_ridge_regression import fit_and_validate_readout

from typing import Dict

class ESNTrainable(tune.Trainable):

    def setup(self, config: Dict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.c_path = None
        self.epochs = None
        self.lr = None

        self.dataset = None
        self.train_loader = None
        self.train_pred_lab = []
        self.Y = None

        self.reservoir: Reservoir = None
        self.readout: torch.Tensor = None

        self.ip_opt: IntrinsicPlasticity = None
        self.score_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()
    
    def step(self):
        self.run_ip_epoch()
        metrics = self.run_readout_epoch()
        return metrics
    
    @torch.no_grad
    def run_ip_epoch(self):
        self.reservoir.train()
        for x, _ in self.train_loader:
            self.reservoir(x.to(self.device))
            self.ip_opt.backward()
            self.ip_opt.step()
            self.reservoir.zero_grad()

    @torch.no_grad
    def run_readout_epoch(self):
        h, y = [], []
        self.reservoir.eval()
        for batch_x, batch_y in self.train_loader:
            h.append(self.reservoir(batch_x.to(self.device)))
            y.append(batch_y)
        
        

    def _build(self):
        pass