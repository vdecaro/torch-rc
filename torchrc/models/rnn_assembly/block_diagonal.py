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
        constrained: Optional[Literal["tanh", "clip", "orthogonal"]] = None,
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
        self._constrained = constrained

        if isinstance(blocks, list):
            self.raw_blocks = blocks
            self._block_sizes = [block.size(0) for block in blocks]
        else:
            if block_sizes is None:
                raise ValueError("block_sizes must be provided if blocks is a function")
            self.raw_blocks = [blocks((b_size, b_size)) for b_size in block_sizes]

        if constrained == "orthogonal":
            for i, block in enumerate(self.raw_blocks):
                setattr(self, f"block_{i}", nn.Parameter(block))
                geotorch.orthogonal(self, f"block_{i}")
        else:
            self._blocks = nn.Parameter(
                block_diagonal(self.raw_blocks),
                requires_grad=constrained is not None,
            )
            self._blocks_mask = nn.Parameter(self._blocks != 0, requires_grad=False)
        if bias:
            self.bias = nn.Parameter(
                torch.normal(mean=0, std=(1 / np.sqrt(self.layer_size)))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = self.blocks
        if self._constrained == "orthogonal":
            blocks = blocks.to(x.device)
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
        if self._constrained != "orthogonal":
            blocks_ = self._blocks * self._blocks_mask
            if self._constrained == "tanh":
                blocks_ = torch.tanh(blocks_)
            elif self._constrained == "clip":
                blocks_ = torch.clamp(blocks_, -0.999, 0.999)
        else:
            blocks_ = block_diagonal(
                [getattr(self, f"block_{i}") for i in range(self.n_blocks)]
            )

        return blocks_
