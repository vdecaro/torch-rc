from typing import Optional, Callable, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Module, Parameter, ModuleList

from . import initializers

__all__ = ['Reservoir']


class Reservoir(Module):
    """
    A Reservoir layer for Echo State Networks
    
    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        num_layers: the number of reservoir layers
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation. Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix. Default: `'normal'`
    """

    def __init__(self,
                 input_size: int, 
                 hidden_size: int,
                 num_layers: int,
                 activation: str,
                 leakage: float = 1.,
                 input_scaling: float = 0.9,
                 rho: float = 0.99,
                 bias: bool = False,
                 kernel_initializer: Union[str, Callable[[Size], Tensor]] = 'uniform',
                 recurrent_initializer: Union[str, Callable[[Size], Tensor]] = 'normal',
                 mode: str = 'vanilla',
                 mu: Optional[float] = None,
                 sigma: Optional[float] = None):
        super().__init__()
        assert num_layers > 0

        self.layers = ModuleList([
            ReservoirLayer(
                input_size=input_size if i == 0 else hidden_size, 
                hidden_size=hidden_size,
                activation=activation,
                leakage=leakage,
                input_scaling=input_scaling,
                rho=rho,
                bias=bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                mode=mode,
                mu=mu,
                sigma=sigma
            )
        for i in range(num_layers)])

    def forward(self, input: Tensor, initial_state: Optional[Union[List[Tensor], Tensor]] = None, 
                mask: Optional[List[Tensor]] = None) -> Tensor:
        """
        Encode input
        :param input: Input graph signal (nodes × input_size)
        :param initial_state: Initial state (nodes × hidden_size) for all reservoir layers, default zeros
        :return: Encoding (samples × dim)
        """
        if initial_state is None:
            initial_state = [torch.zeros(layer.out_features).to(input) for layer in self.layers]
        elif len(initial_state) != self.num_layers and initial_state.dim() == 2:
            initial_state = [initial_state.clone() for _ in range(self.num_layers)]
        
        embeddings = []
        timesteps = input.shape[0]
        input_i = input 
        for i, layer in enumerate(self.layers):
            embeddings_i = []
            for t in range(timesteps):
                h_t = layer(
                        input_i[t],
                        initial_state[i] if t == 0 else h_t,
                        None if mask is None else mask[t]
                      )
                embeddings_i.append(h_t)
            input_i = torch.stack(embeddings_i, dim=0)
            embeddings.append(input_i)
        
        return torch.stack(embeddings, dim=-2)

    @property
    def num_layers(self) -> int:
        """Number of reservoir layers"""
        return len(self.layers)

    @property
    def input_size(self) -> int:
        """Input dimension"""
        return self.layers[0].input_weight.shape[1]

    @property
    def out_features(self) -> int:
        """Embedding dimension"""
        raise NotImplementedError()


class ReservoirLayer(Module):
    """
    A Reservoir layer for Echo State Networks
    
    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation. Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix. Default: `'normal'`
    """

    def __init__(self,
                 input_size: int, 
                 hidden_size: int,
                 activation: str,
                 leakage: float = 1.,
                 input_scaling: float = 0.9,
                 rho: float = 0.99,
                 bias: bool = False,
                 kernel_initializer: Union[str, Callable[[Size], Tensor]] = 'uniform',
                 recurrent_initializer: Union[str, Callable[[Size], Tensor]] = 'normal',
                 mode: str = 'vanilla',
                 mu: Optional[float] = None,
                 sigma: Optional[float] = None) -> None:
        super().__init__()
        assert mode in ['vanilla', 'intrinsic_plasticity']
        if mode == 'intrinsic_plasticity':
            assert mu is not None and sigma is not None
        assert rho < 1 and input_scaling <= 1

        self.input_scaling = Parameter(torch.tensor(input_scaling), requires_grad=False)
        self.rho = Parameter(torch.tensor(rho), requires_grad=False)

        self.W_in = Parameter(
            init_params(kernel_initializer, scale=input_scaling)([hidden_size, input_size]), 
            requires_grad=False
        ) 
        self.W_hat = Parameter(
            init_params(recurrent_initializer, rho=rho)([hidden_size, hidden_size]),
            requires_grad=False
        )
        self.b = Parameter(
            init_params('uniform', scale=input_scaling)(hidden_size), 
            requires_grad=False
        ) if bias else None
        self.f = getattr(torch, activation)

        self.alpha = Parameter(torch.tensor(leakage), requires_grad=False)

        self.mode = mode
        if mode == 'intrinsic_plasticity':
            self.ip_a = Parameter(
                init_params('uniform', scale=0.5)(hidden_size),
                requires_grad=True
            )
            self.ip_b = Parameter(
                init_params('uniform', scale=0.5)(hidden_size),
                requires_grad=True
            )
            self.mu = Parameter(torch.tensor(mu), requires_grad=False)
            self.v = Parameter(torch.tensor(sigma**2), requires_grad=False)


    def forward(self, input: Tensor, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.mode == 'vanilla':
            return self._vanilla_state_comp(input, state, mask)
        if self.mode == 'intrinsic_plasticity':
            return self._ip_state_comp(input, state, mask)


    def _vanilla_state_comp(self, input: Tensor, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h_t = torch.tanh(F.linear(input, self.W_in, self.b) + F.linear(state, self.W_hat))
        output = (1 - self.alpha) * state + self.alpha * h_t
        return output if mask is None else mask * output


    def _ip_state_comp(self, input: Tensor, state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        in_signal = F.linear(input, self.W_in, self.b) + F.linear(state, self.W_hat)
        h_t = torch.tanh(in_signal * self.ip_a + self.ip_b)
        output = (1 - self.alpha) * state + self.alpha * h_t

        if self.training:
            ip_b_grad = -(self.mu/self.v) + (h_t/self.v)*(2*self.v + 1 - h_t*(h_t + self.mu))
            ip_a_grad = 1/self.ip_a + ip_b_grad*in_signal
            if self.ip_b.grad is None:
                self.ip_b.grad = torch.zeros_like(self.ip_b)
                self.ip_a.grad = torch.zeros_like(self.ip_a)
            self.ip_b.grad += ip_b_grad if mask is None else mask * ip_b_grad
            self.ip_a.grad += ip_a_grad if mask is None else mask * ip_a_grad
        
        return output if mask is None else mask * output


    @property
    def input_size(self) -> int:
        """Input dimension"""
        return self.W_in.shape[1]


    @property
    def out_features(self) -> int:
        """Reservoir state dimension"""
        return self.W_hat.shape[1]


    def extra_repr(self) -> str:
        return f'in={self.input_size}, out={self.out_features}, bias={self.bias is not None}'


def init_params(name: str, **options) -> Callable[[Size], Tensor]:
    """
    Gets a random weight initializer
    :param name: Name of the random matrix generator in `esn.initializers`
    :param options: Random matrix generator options
    :return: A random weight generator function
    """
    init = getattr(initializers, name)
    return lambda size: init(size, **options)