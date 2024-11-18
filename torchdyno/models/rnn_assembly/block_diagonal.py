from typing import (
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.models.initializers import block_diagonal


class BlockDiagonal(nn.Module):

    def __init__(
        self,
        blocks: List[torch.Tensor],
        bias: bool = False,
        constrained: Optional[Literal["fixed", "tanh", "clip", "orthogonal"]] = None,
    ):
        """Initializes the block diagonal matrix.

        Args:
            blocks (List[torch.Tensor]): list of blocks.
            bias (bool, optional): whether to use bias. Defaults to False.
            constrained (Optional[Literal["fixed", "tanh", "clip", "orthogonal"]], optional):
                type of constraint. Defaults to None.
        """
        super().__init__()
        self._block_sizes = [block.size(0) for block in blocks]
        self._constrained = constrained

        self._blocks = nn.Parameter(
            block_diagonal(blocks),
            requires_grad=constrained != "fixed",
        )
        self._blocks_mask = nn.Parameter(self._blocks != 0, requires_grad=False)
        self._support_eye = torch.eye(self.layer_size)
        if bias:
            self.bias = nn.Parameter(
                torch.normal(
                    mean=0, std=(1 / np.sqrt(self.layer_size)), dtype=self._blocks.dtype
                ),
            )
        else:
            self.bias = None
        self._cached_blocks = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.blocks, self.bias)

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
        if self._cached_blocks is None or (
            self.training and self._constrained != "fixed"
        ):
            if self._constrained != "orthogonal":
                blocks_ = self._blocks * self._blocks_mask
                if self._constrained == "tanh":
                    blocks_ = torch.tanh(blocks_)
                elif self._constrained == "clip":
                    blocks_ = torch.clamp(blocks_, -0.999, 0.999)
            else:
                symm = self._blocks - self._blocks.T
                blocks_ = self._support_eye + symm + (symm @ symm) / 2
            self._cached_blocks = blocks_

        return self._cached_blocks
