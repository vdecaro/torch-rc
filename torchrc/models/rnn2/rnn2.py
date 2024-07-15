from typing import Callable, Literal, Optional, Tuple
import torch
from torch import nn

from torchrc.utils.initializers import block_diagonal


class RNN2Layer(nn.Module):

    def __init__(
        self,
        input_size: int,
        block_size: int,
        n_blocks: int,
        block_init_fn: Callable[[torch.Size], torch.Tensor],
        block_scaling: Optional[
            Literal["spectral", "singular", "norm", "linear"]
        ] = None,
        block_scaling_value: Optional[float] = None,
        corr_blocks_list: Optional[Tuple[int, int, torch.Tensor]] = None,
        corr_block_init_fn: Optional[Callable[[torch.Size], torch.Tensor]] = None,
        corr_block_scaling: Optional[
            Literal["spectral", "singular", "norm", "linear"]
        ] = None,
        corr_block_scaling_value: Optional[float] = None,
        antisymmetric_mixin: bool = True,
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
            corr_blocks_list (Optional[Tuple[int, int, torch.Tensor]], optional): the \
                list of blocks correlating blocks in the diagonal. Defaults to None.
            corr_block_init_fn (Optional[Callable[[torch.Size], torch.Tensor]], optional): \
                the function to initialize the correlated blocks. Defaults to None.
            corr_block_scaling (Optional[Literal["spectral", "singular", "norm", "linear"]], optional): \
                the type of scaling for the correlated blocks. Defaults to None.
            corr_block_scaling_value (Optional[float], optional): \
                the value for the scaling of the correlated blocks. Defaults to None.
            antisymmetric_mixin (bool, optional): whether to use antisymmetric mixin \
                between RNNs. Defaults to True.
        """
        super().__init__()

        self._block_size = block_size
        self._n_blocks = n_blocks
        self._block_init_fn = block_init_fn
        self._block_scaling = block_scaling
        self._block_scale_value = block_scaling_value

        self._corr_blocks_list = corr_blocks_list
        self._corr_block_init_fn = corr_block_init_fn
        self._corr_block_scaling = corr_block_scaling
        self._corr_block_scale_value = corr_block_scaling_value
        self._antisymmetric_mixin = antisymmetric_mixin

        blocks, mixin = block_diagonal(
            [block_init_fn(block_size) for _ in range(n_blocks)],
            [corr_block_init_fn(block_size) for _ in range(n_blocks)],
            antisymmetric_mixin=antisymmetric_mixin,
            decompose=True,
        )

        self._input_mat = nn.Parameter(
            torch.empty(block_size * n_blocks, input_size), requires_grad=False
        )
        self._blocks = nn.Parameter(blocks, requires_grad=False)
        self._mixin = nn.Parameter(mixin)

    @torch.no_grad()
    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self._block_size).to(self._blocks)

        embeddings = []
        state = initial_state
        timesteps = input.shape[0]
        for t in range(timesteps):
            state = (
                -state
                + self._blocks @ torch.tanh(state)
                + self._mixin @ state
                + self._input_mat @ input[t]
            )
            embeddings.append(state if mask is None else mask * state)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings
