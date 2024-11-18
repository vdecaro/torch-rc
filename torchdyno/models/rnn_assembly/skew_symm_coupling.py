import random
from typing import (
    List,
    Literal,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.models.initializers import block_diagonal_coupling


class SkewAntisymmetricCoupling(nn.Module):

    def __init__(
        self,
        block_sizes: List[int],
        coupling_blocks: List[torch.Tensor],
        coupling_topology: List[Tuple[int, int]],
    ):
        """Initializes the skew antisymmetric coupling layer.

        Args:
            block_sizes (List[int]): list of block sizes.
            coupling_blocks (List[torch.Tensor]): list of coupling blocks.
            coupling_topology (List[Tuple[int, int]]): list of coupling topology.
        """
        super().__init__()
        self._block_sizes = block_sizes
        self._coupling_topology = coupling_topology
        if len(coupling_blocks) != len(coupling_topology):
            raise ValueError(
                "The number of coupling blocks must be equal to the number of coupling topologies."
            )

        self._couplings = nn.Parameter(
            torch.tensor(
                block_diagonal_coupling(
                    block_sizes,
                    [
                        (i, j, coupling_blocks[idx])
                        for idx, (i, j) in enumerate(coupling_topology)
                    ],
                )
            ),
        )
        self._couple_mask = nn.Parameter(self._couplings != 0, requires_grad=False)
        self._cached_coupling = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.couplings)

    @property
    def couplings(self) -> torch.Tensor:
        if self._cached_coupling is None or self.training:
            couple_masked: torch.Tensor = self._couple_mask * self._couplings
            self._cached_coupling = couple_masked - couple_masked.T
        return self._cached_coupling


def get_coupling_indices(
    block_sizes: List[int],
    coupling_topology: Union[int, float, Literal["ring"], List[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Returns the coupling indices based on the topology.

    Args:
        block_sizes (List[int]): list of block sizes.
        coupling_topology (Union[int, float, Literal["ring"]]): coupling topology.

    Returns:
        List[Tuple[int, int]]: list of coupling indices.
    """

    if isinstance(coupling_topology, (int, float)):
        coupling_indices = [
            (i, j)
            for i in range(len(block_sizes) - 1)
            for j in range(i + 1, len(block_sizes))
        ]
        if coupling_topology > 0 and coupling_topology <= 1:
            coupling_topology = int(coupling_topology * len(coupling_indices))

        coupling_indices = random.sample(
            coupling_indices, int(min(len(coupling_indices), coupling_topology))
        )
    elif coupling_topology == "ring":
        coupling_indices = [
            (i, (i + 1) % len(block_sizes)) for i in range(len(block_sizes))
        ]
    else:
        coupling_indices = coupling_topology

    return coupling_indices
