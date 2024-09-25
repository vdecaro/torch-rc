from typing import Callable, List, Literal, Optional, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .block_diagonal import BlockDiagonal
from .skew_symm_coupling import SkewAntisymmetricCoupling


class RNNAssembly(nn.Module):

    def __init__(
        self,
        input_size: int,
        out_size: int,
        block_sizes: List[int],
        block_init_fn: Callable[[torch.Size], torch.Tensor],
        coupling_block_init_fn: Optional[Callable[[torch.Size], torch.Tensor]] = None,
        coupling_perc: Union[int, float] = 0.05,
        generalized_coupling: bool = False,
        eul_step: float = 1e-2,
        activation: str = "relu",
        constrained_blocks: Optional[Literal["tanh", "clip", "orthogonal"]] = None,
    ):
        """Initializes the RNN of RNNs layer.

        Args:
            input_size (int): input size.
            out_size (int): output size.
            block_sizes (List[int]): list of block sizes.
            coupling_indices (List[Tuple[int, int]]): list of coupling indices.
            block_init_fn (Callable[[torch.Size], torch.Tensor]): block initialization
                function.
            coupling_block_init_fn (Optional[Callable[[torch.Size], torch.Tensor]], optional):
                coupling block initialization function. Defaults to None.
            eul_step (float, optional): euler step. Defaults to 1e-2.
            activation (str, optional): activation function. Defaults to "tanh".
            adapt_blocks (bool, optional): whether to adapt blocks. Defaults to False.
            squash_blocks (bool, optional): whether to squash blocks. Defaults to False.
        """
        super().__init__()
        self._input_size = input_size
        self._block_sizes = block_sizes
        self._coupling_perc = coupling_perc
        self._eul_step = eul_step
        self._activation = activation

        self._input_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self._input_size, self.hidden_size),
            ),
            requires_grad=True,
        )

        self._blocks = BlockDiagonal(
            blocks=block_init_fn,
            block_sizes=block_sizes,
            constrained=constrained_blocks,
        )

        self._couplings = SkewAntisymmetricCoupling(
            block_sizes=block_sizes,
            coupling_block_init_fn=coupling_block_init_fn,
            coupling_perc=coupling_perc,
            generalized=generalized_coupling,
            diag_block_matrix=self._blocks.blocks if generalized_coupling else None,
        )

        self._out_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, out_size),
            ),
        )

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self._input_mat)

        embeddings = []
        state = initial_state
        timesteps = input.shape[0]
        if self._activation == "selfnorm":
            activ_fn = lambda x: F.normalize(x, p=2, dim=-1)
        else:
            activ_fn = getattr(torch, self._activation)
        for t in range(timesteps):
            state = state + self._eul_step * (
                -state
                + self._blocks(activ_fn(state))
                + self._couplings(state)
                + input[t] @ self._input_mat
            )
            embeddings.append(state if mask is None else mask * state)
        embeddings = torch.stack(embeddings, dim=0)
        output = embeddings @ self._out_mat
        return output, embeddings

    @property
    def hidden_size(self) -> int:
        return sum(self._block_sizes)
