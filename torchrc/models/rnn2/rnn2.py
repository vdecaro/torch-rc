from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchrc.models.initializers import block_diagonal


class RNN2Layer(nn.Module):

    def __init__(
        self,
        input_size: int,
        block_sizes: List[int],
        coupling_indices: List[Tuple[int, int]],
        block_init_fn: Callable[[torch.Size], torch.Tensor],
        coupling_block_init_fn: Optional[Callable[[torch.Size], torch.Tensor]] = None,
        eul_step: float = 1e-2,
        activation: str = "tanh",
    ):
        """RNN2Layer.
        
        Args:
            input_size (int): the number of expected features in the input `x`
            block_size (int): the number of neurons of each RNN block
            n_blocks (int): the number of RNN blocks
            block_init_fn (Callable[[torch.Size], torch.Tensor]): the function to \
                initialize the blocks
            block_scaling (Optional[Literal["spectral", "singular", "norm", "linear"]], optional): \
                the type of scaling for the blocks. Defaults to None.
            block_scaling_value (Optional[float], optional): the value for the scaling.\
                Defaults to None.
            coupling_blocks_list (Optional[Tuple[int, int, torch.Tensor]], optional): the \
                list of blocks correlating blocks in the diagonal. Defaults to None.
            coupling_block_init_fn (Optional[Callable[[torch.Size], torch.Tensor]], optional): \
                the function to initialize the correlated blocks. Defaults to None.
            coupling_block_scaling (Optional[Literal["spectral", "singular", "norm", "linear"]], optional): \
                the type of scaling for the correlated blocks. Defaults to None.
            coupling_block_scaling_value (Optional[float], optional): \
                the value for the scaling of the correlated blocks. Defaults to None.
            antisymmetric_mixin (bool, optional): whether to use antisymmetric mixin \
                between RNNs. Defaults to True.
            eul_step (float, optional): the Euler step for the integration. Defaults to 1e-2.
        """
        super().__init__()

        self._input_size = input_size
        self._block_sizes = block_sizes
        self._coupling_indices = coupling_indices
        self._eul_step = eul_step
        self._activation = activation
        self._block_init_fn = block_init_fn
        self._coupling_block_init_fn = coupling_block_init_fn

        block_mat, coupling_mat = block_diagonal(
            [block_init_fn((b_size, b_size)) for b_size in block_sizes],
            [
                (
                    i,
                    j,
                    coupling_block_init_fn(
                        (self._block_sizes[i], self._block_sizes[j])
                    ),
                )
                for i, j in coupling_indices
            ],
        )

        self._input_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, self._input_size),
            ),
            requires_grad=False,
        )
        self._blocks = nn.Parameter(block_mat, requires_grad=False)
        self._couplings = nn.Parameter(coupling_mat)
        self._couple_mask = self._couplings != 0
        self._activ_fn = getattr(F, activation)

    @torch.no_grad()
    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self._blocks)

        embeddings = []
        state = initial_state
        timesteps = input.shape[0]
        for t in range(timesteps):
            couple_masked = self._couple_mask * self._couplings
            state = state + self._eul_step * (
                -state
                + self._blocks @ self._activ_fn(state)
                + (couple_masked - couple_masked.T) @ state
                + self._input_mat @ input[t]
            )
            embeddings.append(state if mask is None else mask * state)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings

    @property
    def hidden_size(self) -> int:
        return sum(self._block_sizes)
