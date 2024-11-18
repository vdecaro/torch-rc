from copy import deepcopy
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F
from torch import (
    Generator,
    Size,
    Tensor,
)
from torch.nn import (
    Module,
    Parameter,
)

from torchdyno.models import initializers


class Reservoir(Module):
    """A Reservoir of for Echo State Networks.

    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation.
            Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix.
            Default: `'normal'`
        net_gain_and_bias: if ``True``, the network uses additional ``g`` (gain)
            and ``b`` (bias) parameters. Default: ``False``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        net_gain_and_bias: bool = False,
    ):

        super().__init__()
        assert rho < 1 and input_scaling <= 1

        self.input_scaling = Parameter(torch.tensor(input_scaling), requires_grad=False)
        self.rho = Parameter(torch.tensor(rho), requires_grad=False)

        if isinstance(kernel_initializer, str):
            kernel_initializer_ = getattr(initializers, kernel_initializer)
        else:
            kernel_initializer_ = kernel_initializer

        if isinstance(recurrent_initializer, str):
            recurrent_initializer_ = getattr(initializers, recurrent_initializer)
        else:
            recurrent_initializer_ = recurrent_initializer

        self.W_in = Parameter(
            initializers.rescale(
                kernel_initializer_([hidden_size, input_size]), "linear", input_scaling
            ),
            requires_grad=False,
        )
        self.W_hat = Parameter(
            initializers.rescale(
                recurrent_initializer_([hidden_size, hidden_size]), "spectral", rho
            ),
            requires_grad=False,
        )
        self.b = (
            Parameter(
                initializers.uniform([hidden_size], -input_scaling, input_scaling),
                requires_grad=False,
            )
            if bias
            else None
        )
        self.f = getattr(torch, activation)

        self.alpha = Parameter(torch.tensor(leakage), requires_grad=False)

        self.net_gain_and_bias = net_gain_and_bias
        if net_gain_and_bias:
            self.net_a = Parameter(
                initializers.ones((hidden_size,)), requires_grad=True
            )
            self.net_b = Parameter(
                initializers.zeros((hidden_size,)), requires_grad=True
            )

        self._aux_fwd_comp: Optional[Callable[..., Generator]] = None

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        initial_state: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self.W_hat)
        _fwd_comp = (
            self._state_comp if self._aux_fwd_comp is None else self._aux_fwd_comp
        )

        embeddings = torch.stack(
            [state for state in _fwd_comp(input.to(self.W_hat), initial_state, mask)],  # type: ignore[operator]
            dim=0,
        )

        return embeddings

    def _state_comp(
        self, input: Tensor, initial_state: Tensor, mask: Optional[Tensor] = None
    ):
        timesteps = input.shape[0]
        state = initial_state
        for t in range(timesteps):
            in_signal_t = F.linear(
                input[t].to(self.W_in), self.W_in, self.b
            ) + F.linear(state, self.W_hat)
            if self.net_gain_and_bias:
                in_signal_t = in_signal_t * self.net_a + self.net_b
            h_t = torch.tanh(in_signal_t)
            state = (1 - self.alpha) * state + self.alpha * h_t
            yield state if mask is None else mask * state

    def merge_reservoirs(
        self,
        others: Union["Reservoir", List["Reservoir"]],
        joint_scaling: Optional[float] = None,
        coupled: bool = False,
        independent_inputs: bool = False,
    ) -> "Reservoir":
        """Merges two reservoirs into a single reservoir."""
        if self.net_gain_and_bias:
            raise ValueError("Cannot merge reservoirs with net gain and bias")
        if not isinstance(others, list):
            others = [others]

        new_reservoir = deepcopy(self)
        if independent_inputs:
            new_insize = sum([other.input_size for other in others]) + self.input_size
        else:
            new_insize = self.input_size
        new_hsize = sum([other.hidden_size for other in others]) + self.hidden_size

        W_in = torch.zeros(new_hsize, new_insize)
        if independent_inputs:
            W_in[: self.hidden_size, : self.input_size] = self.W_in.data
            curr_inoffset = self.input_size
            curr_hoffset = self.hidden_size
            for other in others:
                W_in[
                    curr_hoffset : curr_hoffset + other.hidden_size,
                    curr_inoffset : curr_inoffset + other.input_size,
                ] = other.W_in.data
                curr_hoffset += other.hidden_size
                curr_inoffset += other.input_size
        else:
            W_in[: self.hidden_size] = self.W_in.data
            curr_offset = self.hidden_size
            for other in others:
                W_in[curr_offset : curr_offset + other.hidden_size] = other.W_in.data
                curr_offset += other.hidden_size

        if coupled:
            W_hat = torch.empty(new_hsize, new_hsize).uniform_(-1, 1)
        else:
            W_hat = torch.zeros(new_hsize, new_hsize)

        W_hat[: self.hidden_size, : self.hidden_size] = self.W_hat.data
        curr_offset = self.hidden_size
        for other in others:
            W_hat[
                curr_offset : curr_offset + other.hidden_size,
                curr_offset : curr_offset + other.hidden_size,
            ] = other.W_hat.data
            curr_offset += other.hidden_size

        if coupled:
            W_hat = initializers.rescale(
                W_hat,
                "spectral",
                joint_scaling if joint_scaling is not None else self.rho.data,
            )

        new_reservoir.b = None
        if any([other.b is not None for other in others]) or self.b is not None:
            new_reservoir.b = Parameter(torch.zeros(new_hsize))
            if self.b is not None:
                new_reservoir.b.data[: self.hidden_size] = self.b.data
            curr_offset = self.hidden_size
            for other in others:
                if other.b is not None:
                    new_reservoir.b.data[
                        curr_offset : curr_offset + other.hidden_size
                    ] = other.b.data
                curr_offset += other.hidden_size

        new_reservoir.W_in.data = W_in
        new_reservoir.W_hat.data = W_hat
        return new_reservoir

    @property
    def input_size(self) -> int:
        """Input dimension."""
        return self.W_in.shape[1]

    @property
    def hidden_size(self) -> int:
        """Reservoir state dimension."""
        return self.W_hat.shape[1]
