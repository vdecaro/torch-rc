import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Optional
from types import MethodType

from ..model.reservoir import Reservoir


class IntrinsicPlasticity:
    def __init__(self, learning_rate: float, mu: float, sigma: float) -> None:
        self._lr: float = learning_rate
        self.mu: float = mu
        self.v: float = sigma**2

        self._reservoir: Reservoir = None
        self._old_fwd: Callable = None
        self._opt: torch.optim.SGD = None
        self.compiled: bool = False

        self._tmp_in_signal, self._tmp_h, self._tmp_mask = None, None, None

    def step(self) -> None:
        self._reservoir.net_a.grad = F.normalize(self._reservoir.net_a.grad, dim=0)
        self._reservoir.net_b.grad = F.normalize(self._reservoir.net_b.grad, dim=0)
        self._opt.step()

    @torch.no_grad()
    def backward(self):
        net_b_grad = -(self.mu / self.v) + (self._tmp_h / self.v) * (
            2 * self.v + 1 - self._tmp_h**2 + self.mu * self._tmp_h
        )
        net_a_grad = -1 / self._reservoir.net_a + net_b_grad * self._tmp_in_signal
        if self._tmp_mask is not None:
            net_b_grad = self._tmp_mask * net_b_grad
            net_a_grad = self._tmp_mask * net_a_grad
        if net_b_grad.dim() > 1:
            net_b_grad = net_b_grad.sum(0)
            net_a_grad = net_a_grad.sum(0)
        if net_b_grad.dim() > 1:
            net_b_grad = net_b_grad.mean(0)
            net_a_grad = net_a_grad.mean(0)

        self._reservoir.net_b.grad = net_b_grad
        self._reservoir.net_a.grad = net_a_grad
        self._tmp_in_signal, self._tmp_h, self._tmp_mask = None, None, None

    def compile(self, reservoir: Reservoir) -> None:
        if not reservoir.net_gain_and_bias:
            raise ValueError(
                "Reservoir needs trainable net_a and net_b for applying Intrinsic",
                "Plasticity. Initialize with net_gain_and_bias=True.",
            )

        self._reservoir = reservoir
        self._reservoir._aux_fwd_comp = self._ip_state_comp()
        self._opt = torch.optim.SGD(self._reservoir.parameters(), self._lr)
        self.compiled = True

    def detach(self):
        if self.compiled:
            self._reservoir._aux_fwd_comp = None
            self._reservoir = None
            self.compiled = False

    def _ip_state_comp(self) -> Callable:
        def _state_comp(
            input: Tensor,
            initial_state: Tensor,
            mask: Optional[Tensor] = None,
        ) -> None:
            res = self._reservoir
            timesteps, in_signal, h, state = input.shape[0], [], [], initial_state
            for t in range(timesteps):
                in_signal_t = F.linear(input[t], res.W_in, res.b) + F.linear(
                    state, res.W_hat
                )
                h_t = torch.tanh(in_signal_t * res.net_a + res.net_b)
                state = (1 - res.alpha) * state + res.alpha * h_t
                yield state if mask is None else mask * state
                in_signal.append(in_signal_t)
                h.append(h_t)

            if res.training:
                self._tmp_in_signal = torch.stack(in_signal, dim=0)
                self._tmp_h = torch.stack(h, dim=0)
                self._tmp_mask = mask

        return _state_comp
