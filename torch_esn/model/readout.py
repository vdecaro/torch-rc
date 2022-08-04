import sys
from typing import Tuple, Iterator, Optional, List, Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

__all__ = ['Readout', 'fit_readout', 'fit_and_validate_readout']


class Readout(Module):
    """
    A linear readout
    Linear model with bias :math:`y = W x + b`, like Linear in Torch.
    """
    weight: Parameter  # (targets × features)
    bias: Parameter  # (targets)

    def __init__(self, num_features: int, num_targets: int):
        """
        New readout
        :param num_features: Number of input features
        :param num_targets: Number of output targets
        """
        super().__init__()
        self.weight = Parameter(Tensor(num_targets, num_features), requires_grad=False)
        self.bias = Parameter(Tensor(num_targets), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def fit(self, data: Union[Iterator[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]],
            regularization: Union[Optional[float], List[float]] = None,
            validate: Optional[Callable[[Tuple[Tensor, Tensor]], float]] = None,
            verbose: bool = False):
        """
        Fit readout to data
        :param data: Dataset of (features, targets) tuples, or single pair
        :param regularization: Ridge regression lambda, or lambda if validation requested
        :param validate: Validation function, if regularization is to be selected
        :param verbose: Whether to print validation info (default false)
        """
        if not hasattr(data, '__next__'):
            data = iter([data])
        if callable(validate):
            self.weight.data, self.bias.data = fit_and_validate_readout(data, regularization, validate, verbose)
        else:
            self.weight.data, self.bias.data = fit_readout(data, regularization)

    @property
    def num_features(self) -> int:
        """
        Input features
        :return: Number of input features
        """
        return self.weight.shape[1]

    @property
    def num_targets(self) -> int:
        """
        Output targets
        :return: Number of output targets
        """
        return self.weight.shape[0]

    def __repr__(self):
        return f'Readout(features={self.num_features}, targets={self.num_targets})'

