import torch

from torch_esn.model.reservoir import Reservoir
from torch_esn.optimization.intrinsic_plasticity import IntrinsicPlasticity
from torch_esn.optimization.ridge_regression import (
    compute_ridge_matrices,
    solve_ab_decomposition,
    validate_readout,
    compress_ridge_matrices,
)

from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader


class ESNWrapper(object):
    def ip_step(
        self,
        loader: DataLoader,
        reservoir: Reservoir,
        mu: float,
        sigma: float,
        eta: float,
        epochs: int = 1,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        reservoir = reservoir.to(device).train()
        ip_opt = IntrinsicPlasticity(eta, mu, sigma)
        ip_opt.compile(reservoir)
        for e in range(epochs):
            for x, _ in loader:
                reservoir(x.to(device))
                ip_opt.backward()
                ip_opt.step()
                reservoir.zero_grad()
        ip_opt.detach()
        return reservoir

    def ridge_step(
        self,
        loader: DataLoader,
        reservoir: Reservoir,
        l2: Optional[List[float]] = None,
        perc_rec: Optional[float] = 1.0,
        alpha: Optional[float] = 1.0,
        prev_A: Optional[torch.Tensor] = None,
        prev_B: Optional[torch.Tensor] = None,
        with_readout: bool = True,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        reservoir = reservoir.to(device).eval()
        A, B = compute_ridge_matrices(loader, reservoir, device=device)
        if perc_rec < 1.0:
            A, B = compress_ridge_matrices(A, B, perc_rec, alpha)
        if prev_A is not None:
            A += prev_A
            B += prev_B

        if with_readout:
            l2 = [l2] if not isinstance(l2, List) else l2
            readout = [solve_ab_decomposition(A, B, curr_l2, device) for curr_l2 in l2]
            return (readout if len(readout) > 1 else readout[0]), A, B
        else:
            return A, B

    def test_likelihood(
        self,
        loader: DataLoader,
        reservoir: Reservoir,
        mu: float,
        sigma: float,
        device: Optional[str] = None,
    ) -> Tuple[float, int]:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        reservoir = reservoir.to(device).eval()
        likelihood = 0
        for x, _ in loader:
            predictions = reservoir(x.to(device))
            x_likelihood = torch.exp(-((predictions - mu) ** 2 / sigma**2))
            likelihood += x_likelihood.mean(dim=-1).mean(dim=0).sum().item()

        return likelihood / len(loader.dataset)

    def test_accuracy(
        self,
        loader: DataLoader,
        readout: Union[torch.Tensor, List[torch.Tensor]],
        reservoir: Reservoir,
        device: Optional[str] = None,
    ) -> Tuple[float, int]:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return validate_readout(
            readout,
            loader,
            acc_fn,
            reservoir.to(device).eval(),
            device,
        )


def acc_fn(y_true, y_pred):
    count = (y_true.argmax(-1) == y_pred.argmax(-1)).sum(0)
    return (count / y_true.size(0)).cpu().item()
