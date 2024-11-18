from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import (
    Tensor,
    nn,
)
from torch.utils.data import DataLoader

from torchdyno.models import initializers
from torchdyno.optim.ridge_regression import (
    fit_and_validate_readout,
    fit_readout,
)

from .block_diagonal import BlockDiagonal
from .skew_symm_coupling import (
    SkewAntisymmetricCoupling,
    get_coupling_indices,
)


class RNNAssembly(nn.Module):

    def __init__(
        self,
        input_size: int,
        out_size: int,
        blocks: List[torch.Tensor],
        coupling_blocks: List[torch.Tensor],
        coupling_topology: List[Tuple[int, int]],
        eul_step: float = 1e-2,
        activation: str = "tanh",
        constrained_blocks: Optional[
            Literal["fixed", "tanh", "clip", "orthogonal"]
        ] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the RNN of RNNs layer.

        Args:
            input_size (int): size of the input.
            out_size (int): size of the output.
            blocks (List[torch.Tensor]): list of blocks.
            coupling_blocks (List[torch.Tensor]): list of coupling blocks.
            coupling_topology (Union[int, float, List[Tuple[int, int]]]): coupling topology.
            eul_step (float, optional): Euler step. Defaults to 1e-2.
            activation (str, optional): activation function. Defaults to "tanh".
            constrained_blocks (Optional[Literal["fixed", "tanh", "clip", "orthogonal"]], optional):
                type of constraint. Defaults to None.
            dtype (torch.dtype, optional): data type. Defaults to torch.float32.
        """
        super().__init__()
        self._input_size = input_size
        self._eul_step = eul_step
        self._activation = activation
        self._dtype = dtype

        self._blocks = BlockDiagonal(
            blocks=blocks,
            constrained=constrained_blocks,
        )

        self._couplings = SkewAntisymmetricCoupling(
            block_sizes=self._blocks.block_sizes,
            coupling_blocks=coupling_blocks,
            coupling_topology=coupling_topology,
        )

        self._input_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self._input_size, self.hidden_size),
                dtype=self._dtype,
            ),
            requires_grad=False,
        )

        self._out_mat = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, out_size),
                dtype=self._dtype,
            ),
        )
        self.activ_fn = getattr(torch, self._activation)

    @staticmethod
    def from_initializers(
        input_size: int,
        out_size: int,
        block_sizes: List[int],
        block_init_fn: Union[str, Callable[[torch.Size], torch.Tensor]],
        coupling_block_init_fn: Union[str, Callable[[torch.Size], torch.Tensor]],
        coupling_topology: Union[int, float, List[Tuple[int, int]], Literal["ring"]],
        eul_step: float = 1e-2,
        activation: str = "tanh",
        constrained_blocks: Optional[
            Literal["fixed", "tanh", "clip", "orthogonal"]
        ] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "RNNAssembly":
        """Create an RNNAssembly from initializers.

        Args:
            input_size (int): size of the input.
            out_size (int): size of the output.
            block_sizes (List[int]): list of block sizes.
            block_init_fn (Union[str, Callable[[torch.Size], torch.Tensor]]): block
                initializer.
            coupling_block_init_fn (Union[str, Callable[[torch.Size], torch.Tensor]]):
                coupling block initializer.
            coupling_topology (Union[int, float, List[Tuple[int, int]], Literal["ring"]]):
                coupling topology.
            eul_step (float, optional): Euler step. Defaults to 1e-2.
            activation (str, optional): activation function. Defaults to "tanh".
            constrained_blocks (Optional[Literal["fixed", "tanh", "clip", "orthogonal"]], optional):
                type of constraint. Defaults to None.
            dtype (torch.dtype, optional): data type. Defaults to torch.float32.
        """

        if isinstance(block_init_fn, str):
            block_init_fn_: Callable = getattr(initializers, block_init_fn)
        else:
            block_init_fn_ = block_init_fn

        if isinstance(coupling_block_init_fn, str):
            coupling_block_init_fn_ = getattr(initializers, coupling_block_init_fn)
        else:
            coupling_block_init_fn_ = coupling_block_init_fn

        blocks = [block_init_fn_((b_size, b_size), dtype) for b_size in block_sizes]
        coupling_indices = get_coupling_indices(block_sizes, coupling_topology)
        coupling_blocks = [
            coupling_block_init_fn_((block_sizes[i], block_sizes[j]), dtype)
            for i, j in coupling_indices
        ]

        return RNNAssembly(
            input_size=input_size,
            out_size=out_size,
            blocks=blocks,
            coupling_blocks=coupling_blocks,
            coupling_topology=coupling_indices,
            eul_step=eul_step,
            activation=activation,
            constrained_blocks=constrained_blocks,
            dtype=dtype,
        )

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self._input_mat)

        states = self.compute_states(input, initial_state, mask)
        output = states @ self._out_mat
        if self._dtype == torch.complex64:
            output = torch.abs(output)
        return output, states

    def compute_states(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        states = []
        state = (
            initial_state
            if initial_state is not None
            else torch.zeros(self.hidden_size, dtype=self._dtype).to(self._input_mat)
        )
        timesteps = input.shape[0]
        for t in range(timesteps):
            state = state + self._eul_step * (
                -state
                + self._blocks(self.activ_fn(state))
                + self._couplings(state)
                + F.linear(input[t], self._input_mat)
            )
            states.append(state if mask is None else mask * state)
        return torch.stack(states, dim=0)

    def fit_readout(
        self,
        train_loader: DataLoader,
        l2_value: Union[float, List[float]] = 1e-9,
        washout: int = 0,
        score_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
        mode: Optional[Literal["min", "max"]] = None,
        eval_on: Optional[Union[Literal["train"], DataLoader]] = None,
    ) -> Optional[float]:
        """Fit the readout layer.

        Args:
            train_loader (DataLoader): training data loader.
            l2_values (List[float]): list of L2 regularization values.
            score_fn (Callable[[Tensor, Tensor], float]): scoring function.
            washout (int, optional): the amount of timesteps to skip in the training
                dataset to prepare the internal state of the RNN. Defaults to 0.
            score_fn (Optional[Callable[[Tensor, Tensor], float]], optional): scoring
                function. Defaults to None.
            mode (Optional[Literal["min", "max"]], optional): whether to minimize or
                maximize the score. Defaults to None.
            eval_on (Optional[Union[Literal["train"], DataLoader]], optional): evaluation
                data. Defaults to None.

        Returns:
            Optional[float]: the best score.
        """

        if eval_on:
            if score_fn is None:
                raise ValueError("Score function must be provided for validation.")
            if score_fn is not None and mode is None:
                raise ValueError("Mode must be provided for optimization.")

            if eval_on == "train":
                eval_loader = train_loader
            elif isinstance(eval_on, DataLoader):
                eval_loader = eval_on
            else:
                raise ValueError("Evaluation data must be provided as DataLoader.")

            if not isinstance(l2_value, list):
                l2_value = [l2_value]

            readout, best_l2, best_score = fit_and_validate_readout(
                train_loader=train_loader,
                eval_loader=eval_loader,
                l2_values=l2_value,
                preprocess_fn=self.compute_states,
                skip_first_n=washout,
                score_fn=score_fn,
                mode=mode,
                device=next(self.parameters()).device,
            )

        else:
            readout = fit_readout(
                train_loader,
                preprocess_fn=self.compute_states,
                skip_first_n=washout,
                l2=l2_value,
                device=next(self.parameters()).device,
            )

        self._out_mat.data = readout

        if eval_on:
            return best_score
        return None

    @property
    def hidden_size(self) -> int:
        return self._blocks.layer_size
