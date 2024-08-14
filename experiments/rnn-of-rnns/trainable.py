from typing import Dict
from ray import tune
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .utils import load_mnist

from torchrc.models.rnn2 import RNN2Layer


class RNNofRNNsTrainable(tune.Trainable):

    def setup(self, config: Dict):
        self.train_loader, self.eval_loader = load_mnist(128, 1024)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.build_trial(config)

    def step(self):
        run_loss_t = 0.0
        run_acc_t = 0.0
        n_samples = 0
        self.model.train()
        for x, y in self.train_loader:
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            all_pred, _ = self.model(x)
            y_pred = all_pred[-1]
            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()

            run_loss_t = (run_loss_t * n_samples) + (loss.item() * x.size(1))
            batch_acc = (y_pred.argmax(dim=-1) == y).sum().item()
            run_acc_t = (run_acc_t * n_samples) + (batch_acc * x.size(1))
            n_samples += y.size(0)
            run_loss_t /= n_samples
            run_acc_t /= n_samples
        self.lr.step()

        self.model.eval()
        run_loss_e = 0.0
        run_acc_e = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_pred, _ = self.model(x)
                y_pred = all_pred[-1]
                loss = self.loss(y_pred, y)
                run_loss_e = (run_loss_e * n_samples) + (loss.item() * x.size(1))
                batch_acc = (y_pred.argmax(dim=-1) == y).sum().item()
                run_acc_e = (run_acc_e * n_samples) + (batch_acc * x.size(1))
                n_samples += y.size(0)
                run_loss_e /= n_samples
                run_acc_e /= n_samples
        return {
            "train_loss": run_loss_t,
            "train_acc": run_acc_t,
            "test_loss": run_loss_e,
            "test_acc": run_acc_e,
        }

    def save_checkpoint(self, checkpoint_dir: str) -> Dict | None:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict()}

    def load_checkpoint(self, checkpoint: Dict):
        self.model.load_state_dict(checkpoint["model"])
        self.opt.load_state_dict(checkpoint["opt"])

    def build_trial(self, config):
        # Build model with configuration
        model_config = config["rnn_params"]
        self.model = RNN2Layer(
            input_size=1,
            out_size=10,
            block_sizes=model_config["blocks"],
            coupling_indices=model_config["couplings"],
            block_init_fn=get_init_fn(*model_config["block_init_fn"]),
            coupling_block_init_fn=get_init_fn(*model_config["couple_init_fn"]),
            eul_step=model_config["eul_step"],
            activation=model_config["activation"],
            adapt_blocks=model_config["adapt_blocks"],
            squash_blocks=model_config["squash_blocks"],
            orthogonal_blocks=model_config["orthogonal_blocks"],
        )
        self.model.to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.opt = Adam(self.model.parameters(), lr=config["opt_params"]["lr"])
        self.lr = MultiStepLR(
            self.opt,
            milestones=config["opt_params"]["decay_epochs"],
            gamma=config["opt_params"]["decay_scalar"],
        )

    def reset_config(self, new_config):
        self.build_trial(new_config)


def get_init_fn(fn_str: str, *args):
    from torchrc.models.initializers import sparse, diagonal, orthogonal

    if fn_str == "sparse":
        return lambda x: sparse(x, *args)
    if fn_str == "orthogonal":
        return lambda x: orthogonal(x, *args)
    if fn_str == "diagonal":
        return lambda x: diagonal(x)
    raise ValueError(f"Unknown initialization function {fn_str}")
