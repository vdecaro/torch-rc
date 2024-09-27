from typing import Callable, List, Literal, Optional, Union
import numpy as np
import torch
from torch import Size, Tensor, nn
from torch.utils.data import DataLoader

from torchrc.optim.ridge_regression import fit_and_validate_readout, fit_readout
from .reservoir import Reservoir


class EchoStateNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        layers: List[int],
        output_size: int,
        arch_type: Literal["stacked", "multi"] = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "normal",
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
            recurrent_initializer (Union[str, Callable[[Size], Tensor]], optional): recurrent initializer. Defaults to "normal".
            net_gain_and_bias (bool, optional): whether to use net gain and bias. Defaults to False.
        """
        if len(layers) == 0:
            raise ValueError("At least one hidden layer must be defined.")
        if len(layers) > 1 and net_gain_and_bias:
            raise ValueError(
                "Net gain and bias can only be used with one hidden layer."
            )
        if arch_type not in ["stacked", "multi"]:
            raise ValueError(
                f"Unknown architecture type: {arch_type}. Choose from 'stacked' or 'multi'."
            )
        super().__init__()
        self._arch_type = arch_type
        self.reservoirs = nn.ModuleList(
            [
                Reservoir(
                    input_size if i == 0 else layers[i - 1],
                    layers[i],
                    activation,
                    leakage,
                    input_scaling,
                    rho,
                    bias,
                    kernel_initializer,
                    recurrent_initializer,
                    net_gain_and_bias,
                )
                for i in range(len(layers))
            ]
        )

        self.readout = nn.Parameter(
            torch.normal(
                mean=0,
                std=1 / np.sqrt(self.state_dim),
                size=(self.state_dim, output_size),
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
        states = []
        for i, reservoir in enumerate(self.reservoirs):
            x = reservoir(x, initial_state[i] if initial_state else None)
            states.append(x)

        if self._arch_type == "multi":
            x = torch.cat(states, dim=-1)

        pred = x @ self.readout

        if return_states:
            return pred, states
        return pred

    def fit_readout(
        self,
        train_loader: DataLoader,
        l2_value: Union[float, List[float]] = 1e-6,
        score_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
        mode: Optional[Literal["min", "max"]] = None,
        eval_on: Optional[Union[Literal["train"], DataLoader]] = None,
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

        def preprocess_fn(x):
            if self._arch_type == "multi":
                states = []
                for _, reservoir in enumerate(self.reservoirs):
                    states.append(reservoir(x))
                return torch.cat(states, dim=-1)
            elif self._arch_type == "stacked":
                state = x
                for reservoir in self.reservoirs:
                    state = reservoir(state)
                return state

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
                preprocess_fn=preprocess_fn,
                score_fn=score_fn,
                mode=mode,
                device=next(self.parameters()).device,
            )
            print(f"ESN Trained. \n\Chosen L2: {best_l2}\n\Score: {best_score:.4f}")

        else:
            readout = fit_readout(
                train_loader,
                preprocess_fn=preprocess_fn,
                l2=l2_value,
                device=next(self.parameters()).device,
            )
            print("ESN Trained.")

        self.readout.data = readout.T

    @property
    def state_dim(self) -> int:
        """Returns the size of the hidden state."""
        if self._arch_type == "stacked":
            return self.reservoirs[-1].hidden_size
        elif self._arch_type == "multi":
            return sum([reservoir.hidden_size for reservoir in self.reservoirs])
