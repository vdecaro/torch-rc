import random
from typing import Callable, List, Optional, Union
import numpy as np
import scipy
import torch
from torch import nn

from torchrc.models.initializers import block_diagonal_coupling


class SkewAntisymmetricCoupling(nn.Module):

    def __init__(
        self,
        block_sizes: List[int],
        coupling_block_init_fn: Callable[[torch.Size], torch.Tensor],
        coupling_perc: Union[int, float] = 20,
        generalized: bool = False,
        diag_block_matrix: Optional[torch.Tensor] = None,
    ):
        """Initializes the skew antisymmetric coupling layer.

        Args:
            block_sizes (List[int]): list of block sizes.
            coupling_block_init_fn (Callable[[torch.Size], torch.Tensor]): coupling block initialization function.
            coupling_perc (Union[int, float], optional): percentage of couplings. Defaults to 20.
            generalized (bool, optional): whether to use generalized skew antisymmetric coupling. Defaults to False.
            diag_block_matrix (Optional[torch.Tensor], optional): diagonal block matrix to compute the metric
                of the generalized skew antysimmetric matrix. Defaults to None.
        """
        super().__init__()
        if coupling_perc < 0 or coupling_perc > (
            len(block_sizes) * (len(block_sizes) - 1) / 2
        ):
            raise ValueError(
                f"coupling_perc must be either a percentage in (0,1) or an integer in [1, len(block_sizes)]. Got {coupling_perc}"
            )
        if generalized and diag_block_matrix is None:
            raise ValueError(
                "diag_block_matrix must be provided if generalized is True"
            )

        self.block_sizes = block_sizes
        self.generalized = generalized
        if isinstance(coupling_perc, float):
            coupling_perc = round(coupling_perc * len(block_sizes))
        coupling_indices = [(i, j) for i in range(16) for j in range(16) if i < j]
        random.shuffle(coupling_indices)
        coupling_indices = coupling_indices[:coupling_perc]
        coupling_matrices = [
            (i, j, coupling_block_init_fn((block_sizes[i], block_sizes[j])))
            for i, j in coupling_indices
        ]

        self._couplings = nn.Parameter(
            torch.tensor(block_diagonal_coupling(block_sizes, coupling_matrices))
        )
        self._couple_mask = nn.Parameter(self._couplings != 0, requires_grad=False)

        if generalized:
            self._metric_mat = nn.Parameter(
                _build_metric_matrix(diag_block_matrix), requires_grad=False
            )
            self._metric_mat_inv = nn.Parameter(
                torch.inverse(self._metric_mat), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        couple_masked: torch.Tensor = self._couple_mask * self._couplings
        if self.generalized:
            transform = (
                couple_masked.T
                - (self._metric_mat @ couple_masked) @ self._metric_mat_inv
            )
        else:
            transform = couple_masked - couple_masked.T

        return x @ transform


def _build_metric_matrix(diag_block_matrix: torch.Tensor) -> torch.Tensor:
    # what we actually want to find metric for is W - I -> just W won't be stable here
    # also first need to focus on abs(W), not W itself! that is what linear stable test can find, the same metric will then work for the other (per Thm 1)
    full_W = diag_block_matrix.detach().numpy()
    W = np.abs(
        diag_block_matrix
    )  # diagonal is set to 0 already so no need to worry about that
    W = W - np.identity(W.shape[0])
    # this just finds some M that will work, could be many others
    Q = np.identity(W.shape[0])
    # solve for M in -Q = M * W + np.transpose(W) * M
    # using integration formula for LTI system
    P = np.zeros(W.shape)
    for i in range(W.shape[0]):
        # integrating elementwise
        # keep off-diags as 0 to save time with larger martrix, as know there will be some diagonal metric, expect good odds that will find one with Q = I
        # will confirm the metric works before moving forward though (done in final function below), to be sure with stability guarantee
        def func_to_integrate(t):
            og_func = np.exp(np.transpose(W) * t) * Q * np.exp(W * t)
            return og_func[i, i]

        P[i, i] = scipy.integrate.quad(func_to_integrate, 0, np.inf)[0]
    if np.max(np.linalg.eigvals(P)) <= 0:
        # guaranteed M will be symmetric as it is definitely diagonal here, but also need it be PD for it to be a valid metric
        # should never reach this in theory, but add as a check to be safe
        print("returned metric not PD, problem!")
        return None

    check_formula = (
        P * (np.abs(full_W) - np.identity(full_W.shape[0]))
        + np.transpose(np.abs(full_W) - np.identity(full_W.shape[0])) * P
    )
    if np.max(np.linalg.eigvals(check_formula)) >= 0:
        print("problem with found metric!")
        return None
    return P
