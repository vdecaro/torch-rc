import torch

from torch.utils.data import DataLoader
from torch_esn.data.datasets import get_dataset
from torch_esn.data.util.seq_loader import seq_collate_fn

from .base import ESNWrapper
from typing import List, Optional, Tuple, Union
from torch_esn.model.reservoir import Reservoir


class VanillaESNWrapper(ESNWrapper):
    def __init__(self, dataset: str, users: List[str], batch_size: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.dataset = get_dataset(dataset, users)

    def ip_step(
        self,
        reservoir: Reservoir,
        mu: float,
        sigma: float,
        eta: float,
        epochs: int = 1,
        device: Optional[str] = None,
    ):
        return super().ip_step(
            self.get_loader(),
            reservoir,
            mu=mu,
            sigma=sigma,
            eta=eta,
            epochs=epochs,
            device=device,
        )

    def ridge_step(
        self,
        reservoir: Reservoir,
        l2: Optional[List[float]] = None,
        perc_rec: Optional[float] = 1.0,
        alpha: Optional[float] = 1.0,
        prev_A: Optional[torch.Tensor] = None,
        prev_B: Optional[torch.Tensor] = None,
        with_readout: bool = True,
        device: Optional[str] = None,
    ):
        return super().ridge_step(
            self.get_loader(),
            reservoir,
            l2=l2,
            perc_rec=perc_rec,
            alpha=alpha,
            prev_A=prev_A,
            prev_B=prev_B,
            with_readout=with_readout,
            device=device,
        )

    def test_likelihood(
        self,
        reservoir: Reservoir,
        mu: float,
        sigma: float,
        device: Optional[str] = None,
    ) -> Tuple[float, int]:
        return super().test_likelihood(
            loader=self.get_loader(),
            reservoir=reservoir,
            mu=mu,
            sigma=sigma,
            device=device,
        )

    def test_accuracy(
        self,
        readout: Union[torch.Tensor, List[torch.Tensor]],
        reservoir: Reservoir,
        device: Optional[str] = None,
    ) -> Tuple[float, int]:
        return super().test_accuracy(
            loader=self.get_loader(),
            readout=readout,
            reservoir=reservoir,
            device=device,
        )

    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=seq_collate_fn(),
        )

    def get_dataset_size(self):
        return len(self.dataset)
