from torch import nn

from .rnn_layer import RNNLayer


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, activation="tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [RNNLayer(input_size, hidden_size, activation) for _ in range(num_layers)]
        )

    def forward(self, x, h=None):
        if h is None:
            h = [None] * self.num_layers

        for layer, state in zip(self.layers, h):
            x = layer(x, state)

        return x
