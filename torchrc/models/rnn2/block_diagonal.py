from typing import Callable, List, Literal, Optional, Union
import numpy as np
import torch
from torch import nn
import geotorch

from torchrc.models.initializers import block_diagonal


class BlockDiagonal(nn.Module):

    def __init__(
        self,
        blocks: Union[Callable[[torch.Size], torch.Tensor], List[torch.Tensor]],
        block_sizes: Optional[int] = None,
        bias: bool = False,
        orthogonal: bool = False,
        requires_grad: bool = True,
        squash_blocks: Optional[Literal["tanh", "clip"]] = None,
    ):
        """Initializes the block diagonal matrix.

        Args:
            blocks (Union[Callable[[torch.Size], torch.Tensor], List[torch.Tensor]]): blocks
                of the block diagonal matrix.
            block_sizes (Optional[int], optional): block sizes. Defaults to None.
            bias (bool, optional): whether to use bias. Defaults to False.
            orthogonal (bool, optional): whether to use orthogonal initialization. Defaults to False.
            requires_grad (bool, optional): whether to require gradients. Defaults to True.
            squash_blocks (Optional[Literal["tanh", "clip"]], optional): whether to squash blocks.
                Defaults to None.
        """
        super().__init__()
        self._block_sizes = block_sizes
        self._orthogonal = orthogonal
        self.squash_blocks = squash_blocks

        if isinstance(blocks, list):
            self.raw_blocks = blocks
            self._block_sizes = [block.size(0) for block in blocks]
        else:
            if block_sizes is None:
                raise ValueError("block_sizes must be provided if blocks is a function")
            self.raw_blocks = [blocks(b_size) for b_size in block_sizes]

        if orthogonal:
            for i, block in enumerate(self.raw_blocks):
                setattr(self, f"block_{i}", nn.Parameter(block))
                geotorch.orthogonal(self, f"block_{i}")
        else:

            self._blocks = nn.Parameter(
                block_diagonal(self.raw_blocks), requires_grad=requires_grad
            )
            self.register_buffer("_blocks_mask", (self._blocks_mask != 0))
        if bias:
            self.bias = nn.Parameter(
                torch.normal(mean=0, std=(1 / np.sqrt(self.layer_size)))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = self.blocks
        if self._orthogonal:
            transform = torch.cat([x @ block for block in blocks], dim=-1)
        else:
            transform = x @ blocks
        if hasattr(self, "bias"):
            transform = transform + self.bias
        return transform

    @property
    def n_blocks(self) -> int:
        return len(self._block_sizes)

    @property
    def block_sizes(self) -> List[int]:
        return self._block_sizes

    @property
    def layer_size(self) -> int:
        return sum(self._block_sizes)

    @property
    def blocks(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self._orthogonal:
            blocks_ = self._blocks * self._blocks_mask
            if self.squash_blocks == "tanh":
                blocks_ = torch.tanh(blocks_)
            elif self.squash_blocks == "clip":
                blocks_ = torch.clamp(blocks_, -1, 1)
            return blocks_
        blocks_ = [getattr(self, f"block_{i}") for i in range(self.n_blocks)]
        for i, block in enumerate(blocks_):
            if self.squash_blocks == "tanh":
                blocks_[i] = torch.tanh(block)
            elif self.squash_blocks == "clip":
                blocks_[i] = torch.clamp(block, -1, 1)
        return blocks_
