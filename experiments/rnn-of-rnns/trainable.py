from typing import Dict
from ray import tune
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from utils import load_mnist

from torchrc.models.rnn_assembly import RNNAssembly


class RNNAssemblyTrainable(tune.Trainable):

    def setup(self, config: Dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.build_trial(config)

    def step(self):
        run_loss_t = 0.0
        run_acc_t = 0.0
        n_batches = 0
        self.model.train()
        for x, y in self.train_loader:
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            all_pred, _ = self.model(x)
            y_pred = all_pred[-1]
            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()

            run_loss_t += loss.item()
            run_acc_t += (
                (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
            ) / len(y)
            n_batches += 1
        run_loss_t /= n_batches
        run_acc_t /= n_batches
        self.lr.step()

        self.model.eval()
        run_loss_e = 0.0
        run_acc_e = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_pred, _ = self.model(x)
                y_pred = all_pred[-1]
                loss = self.loss(y_pred, y)
                run_loss_e += loss.item()
                run_acc_e += (
                    (y_pred.argmax(dim=-1).flatten() == y.flatten()).sum().item()
                ) / len(y)
                n_batches += 1
        run_loss_e /= n_batches
        run_acc_e /= n_batches
        res = {
            "train_loss": run_loss_t,
            "train_acc": run_acc_t,
            "test_loss": run_loss_e,
            "test_acc": run_acc_e,
        }
        return res

    def save_checkpoint(self, checkpoint_dir: str) -> Dict | None:
        self.model.to("cpu")
        torch.save(self.model, f"{checkpoint_dir}/model.pth")
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint: Dict):
        self.model = torch.load(checkpoint["model.pth"])
        self.model.to(self.device)

    def build_trial(self, config):
        # Build model with configuration

        self.train_loader, self.eval_loader = load_mnist(
            128, 1024, permute_seed=config["permute_seed"], root=config["root"]
        )
        block_config = get_block_config(config["block_config"])
        self.model = RNNAssembly(
            input_size=1,
            out_size=10,
            block_sizes=config["block_sizes"],
            coupling_perc=config["coupling_perc"],
            block_init_fn=get_init_fn(*block_config[0]),
            coupling_block_init_fn=get_init_fn(*config["coupling_block_init_fn"]),
            generalized_coupling=config["generalized_coupling"],
            eul_step=config["eul_step"],
            activation=config["activation"],
            constrained_blocks=block_config[1],
        )
        self.model.to(device=self.device)

        self.loss = nn.CrossEntropyLoss()
        self.opt = Adam(self.model.parameters())
        self.lr = MultiStepLR(
            self.opt,
            milestones=config["decay_epochs"],
            gamma=config["decay_scalar"],
        )


def get_init_fn(fn_str: str, *args):
    from torchrc.models.initializers import sparse, diagonal, orthogonal

    if fn_str == "sparse":
        return lambda x: sparse(x, *args)
    if fn_str == "orthogonal":
        return lambda x: orthogonal(x)
    if fn_str == "diagonal":
        return lambda x: diagonal(x)
    raise ValueError(f"Unknown initialization function {fn_str}")


def get_block_config(idx: int):
    # if idx == 0:
    #     return (("orthogonal",), "orthogonal")
    if idx == 1:
        return (("sparse", 0.03), None)
    if idx == 2:
        return (("sparse", 0.1), None)
    if idx == 3:
        return (("diagonal",), "tanh")
    if idx == 4:
        return (("diagonal",), "clip")
