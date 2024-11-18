from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Union,
    overload,
)

import numpy as np
import torch
from torch import (
    Size,
    Tensor,
    nn,
)
from torch.utils.data import DataLoader

from torchdyno.optim.ridge_regression import (
    fit_and_validate_readout,
    fit_readout,
    solve_ab_decomposition,
    validate_readout,
)

from .reservoir import Reservoir


class EchoStateNetwork(nn.Module):

    @overload
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_sizes: List[int],
        arch_type: Literal["stacked", "multi", "parallel"] = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        net_gain_and_bias: bool = False,
    ):
        """Initializes the Echo State Network.

        Args:
            input_size (int): size of the input.
            layers (List[int]): list of hidden layer sizes.
            activation (str, optional): activation function. Defaults to "tanh".
            leakage (float, optional): leakage value. Defaults to 1.0.
            input_scaling (float, optional): input scaling value. Defaults to 0.9.
            rho (float, optional): spectral radius. Defaults to 0.99.
            bias (bool, optional): whether to use bias. Defaults to False.
            kernel_initializer (Union[str, Callable[[Size], Tensor]], optional): kernel initializer. Defaults to "uniform".
            recurrent_initializer (Union[str, Callable[[Size], Tensor]], optional): recurrent initializer. Defaults to "uniform".
            net_gain_and_bias (bool, optional): whether to use net gain and bias. Defaults to False.
        """
        ...

    @overload
    def __init__(
        reservoirs: List[Reservoir],
        output_size: int,
        combine_reservoirs: Literal[
            "stacked", "multi", "parallel", "merge"
        ] = "stacked",
        joint_scaling: float = 1.0,
        independent_inputs: bool = False,
    ):
        """Initializes the Echo State Network from a list of reservoirs.

        Args:
            reservoirs (List[Reservoir]): list of reservoirs.
            output_size (int): output size.
            combine_reservoirs (Literal["parallel", "joint"], optional): how to combine
                the reservoirs. Defaults to "stacked".
            joint_scaling (Optional[float], optional): scaling factor for joint
                reservoirs. Defaults to None.
            independent_inputs (bool, optional): whether the inputs are independent.
                Defaults to False.
        """
        ...

    def __init__(  # type: ignore[misc]
        self,
        output_size: int,
        input_size: Optional[int] = None,
        layer_sizes: Optional[List[int]] = None,
        arch_type: Literal["stacked", "multi", "parallel"] = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        net_gain_and_bias: bool = False,
        reservoirs: Optional[List[Reservoir]] = None,
        combine_reservoirs: Literal[
            "stacked", "multi", "parallel", "merge"
        ] = "stacked",
        joint_scaling: float = 1.0,
        independent_inputs: bool = False,
    ):
        super().__init__()
        if input_size is None and reservoirs is None:
            raise ValueError("Either input size or reservoirs must be provided.")
        if output_size is None:
            raise ValueError("Output size must be provided.")

        if reservoirs is not None:
            self._from_reservoirs(
                reservoirs,
                output_size,
                combine_reservoirs,
                joint_scaling,
                independent_inputs,
            )
        else:
            if input_size is None:
                raise ValueError("Input size must be provided.")
            if layer_sizes is None:
                raise ValueError("Layer sizes must be provided.")
            if len(layer_sizes) == 0:
                raise ValueError("At least one hidden layer must be defined.")
            if arch_type not in ["stacked", "multi", "parallel"]:
                raise ValueError(
                    f"Unknown architecture type: {arch_type}. Choose from 'stacked', 'multi' or 'parallel'."
                )
            self._from_hyperparameters(
                input_size,
                output_size,
                layer_sizes,
                arch_type,
                activation,
                leakage,
                input_scaling,
                rho,
                bias,
                kernel_initializer,
                recurrent_initializer,
                net_gain_and_bias,
            )

    def _from_hyperparameters(
        self,
        input_size: int,
        output_size: int,
        layer_sizes: List[int],
        arch_type: Literal["stacked", "multi", "parallel"] = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        net_gain_and_bias: bool = False,
    ):

        if layer_sizes is None or len(layer_sizes) == 0:
            raise ValueError("At least one hidden layer must be defined.")
        if len(layer_sizes) > 1 and net_gain_and_bias:
            raise ValueError(
                "Net gain and bias can only be used with one hidden layer."
            )
        if arch_type not in ["stacked", "multi", "parallel"]:
            raise ValueError(
                f"Unknown architecture type: {arch_type}. Choose from 'stacked', 'multi' or 'parallel'."
            )
        self.arch_type = arch_type
        self.reservoirs = nn.ModuleList(
            [
                Reservoir(
                    input_size if i == 0 else layer_sizes[i - 1],
                    layer_sizes[i],
                    activation,
                    leakage,
                    input_scaling,
                    rho,
                    bias,
                    kernel_initializer,
                    recurrent_initializer,
                    net_gain_and_bias,
                )
                for i in range(len(layer_sizes))
            ]
        )
        self.readout = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, output_size),
            ),
            requires_grad=True,
        )
        self.ridge_A, self.ridge_B = None, None

    def _from_reservoirs(
        self,
        reservoirs: List[Reservoir],
        output_size: int,
        combine_reservoirs: Literal[
            "stacked", "multi", "parallel", "merge"
        ] = "stacked",
        joint_scaling: Optional[float] = None,
        independent_inputs: bool = False,
    ):
        """Initialize the network from a list of reservoirs.

        Args:
            reservoirs (List[Reservoir]): list of reservoirs.
            output_size (int): output size.
            combine_reservoirs (Literal["parallel", "joint"], optional): how to combine
                the reservoirs. Defaults to "stacked".
            joint_scaling (Optional[float], optional): scaling factor for joint
                reservoirs. Defaults to None.
            independent_inputs (bool, optional): whether the inputs are independent.
                Defaults to False.
        """
        if combine_reservoirs in ["parallel", "merge"]:
            self.arch_type = "stacked"
            reservoir, others = reservoirs[0], reservoirs[1:]
            reservoir = reservoir.merge_reservoirs(
                others,
                joint_scaling=joint_scaling,
                coupled=combine_reservoirs == "merge",
                independent_inputs=independent_inputs,
            )
            reservoirs = [reservoir]
        else:
            self.arch_type = combine_reservoirs  # type: ignore[assignment]
        self.reservoirs = nn.ModuleList(reservoirs)
        self.readout = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, output_size),
            ),
            requires_grad=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): input tensor.
            initial_state (List[torch.Tensor]): initial state of the reservoirs.
            return_states (bool, optional): whether to return the states. Defaults to False.

        Returns:
            torch.Tensor: output tensor.
        """
        if initial_state is not None and len(initial_state) != len(self.reservoirs):
            raise ValueError(
                f"Initial state must be provided for each reservoir. Expected "
                f"{len(self.reservoirs)}, got {len(initial_state)}."
            )

        x, states = self.apply_reservoirs(x, initial_state)
        pred = x @ self.readout

        if return_states:
            return pred, states
        return pred

    def apply_reservoirs(
        self, x: torch.Tensor, initial_state: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply the reservoirs to the input.

        Args:
            x (torch.Tensor): input tensor.
            initial_state (Optional[torch.Tensor]): initial state of the reservoirs.

        Returns:
            torch.Tensor: output tensor.
        """
        states = []
        for i, reservoir in enumerate(self.reservoirs):
            x = reservoir(x, initial_state[i] if initial_state else None)
            states.append(x)

        if self.arch_type in ["multi", "parallel"]:
            x = torch.cat(states, dim=-1)

        return x, states

    def fit_readout(
        self,
        train_loader: DataLoader,
        l2_value: Union[float, List[float]] = 1e-9,
        washout: int = 0,
        score_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
        mode: Optional[Literal["min", "max"]] = None,
        eval_on: Optional[Union[Literal["train"], DataLoader]] = None,
        store_matrices: bool = False,
    ):
        """Fit the readout layer.

        Args:
            train_loader (DataLoader): training data loader.
            eval_loader (DataLoader): evaluation data loader.
            l2_values (List[float]): list of L2 regularization values.
            score_fn (Callable[[Tensor, Tensor], float]): scoring function.
            mode (Literal["min", "max"]): optimization mode.
            weights (Optional[List[float]], optional): weights for the loss. Defaults to None.
            preprocess_fn (Optional[Callable], optional): preprocessing function. Defaults to None.
            device (Optional[str], optional): device to use. Defaults to None.
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

            readout, best_l2, best_score, A, B = fit_and_validate_readout(
                train_loader=train_loader,
                eval_loader=eval_loader,
                l2_values=l2_value,
                preprocess_fn=lambda x: _preprocess_fn(self, x),
                skip_first_n=washout,
                score_fn=score_fn,
                mode=mode,
                device=next(self.parameters()).device,
            )

        else:
            readout, A, B = fit_readout(
                train_loader,
                preprocess_fn=lambda x: _preprocess_fn(self, x),
                skip_first_n=washout,
                l2=l2_value,
                device=next(self.parameters()).device,
            )

        self.readout.data = readout
        if store_matrices:
            self.ridge_A, self.ridge_B = A, B
        if eval_on:
            return best_score

    def fit_from_matrices(self, l2: Optional[float] = None):
        """Fit the readout layer from the stored matrices."""

        if self.ridge_A is None or self.ridge_B is None:
            raise ValueError("Matrices have not been stored.")

        self.readout.data = solve_ab_decomposition(
            A=self.ridge_A, B=self.ridge_B, l2=l2, device=next(self.parameters()).device
        )

    def evaluate(
        self,
        test_loader: DataLoader,
        score_fn: Callable[[Tensor, Tensor], float],
        washout: int = 0,
    ) -> float:
        """Evaluate the model.

        Args:
            test_loader (DataLoader): test data loader.
            score_fn (Callable[[Tensor, Tensor], float]): scoring function.
            washout (int, optional): washout steps. Defaults to 0.

        Returns:
            float: evaluation score.
        """
        return validate_readout(
            self.readout,
            test_loader,
            preprocess_fn=lambda x: _preprocess_fn(self, x),
            score_fn=score_fn,
            skip_first_n=washout,
            device=next(self.parameters()).device,
        )

    @property
    def hidden_size(self) -> int:
        """Returns the size of the hidden state."""
        if self.arch_type == "stacked":
            return self.reservoirs[-1].hidden_size
        else:
            return sum([reservoir.hidden_size for reservoir in self.reservoirs])


def _preprocess_fn(esn: EchoStateNetwork, x: torch.Tensor) -> torch.Tensor:
    states = []
    for reservoir in esn.reservoirs:
        x = reservoir(x)
        states.append(x)

    if esn.arch_type in ["multi", "parallel"]:
        return torch.cat(states, dim=-1)
    else:
        return states[-1]
