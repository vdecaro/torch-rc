from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class RNNLayer(nn.Module):
    """Implements the Elman RNN layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_in = nn.Parameter(
            torch.randn(hidden_size, input_size) / hidden_size**0.5
        )

        self.w_rec = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / hidden_size**0.5
        )

        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        if h is None:
            h = torch.zeros(self.hidden_size, device=x.device)

        for t in range(x.size(0)):
            h = x[t] @ self.w_in + h @ self.w_rec + self.bias
            h = torch.tanh(h)

        return h
