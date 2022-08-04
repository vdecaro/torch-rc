import sys
from typing import Tuple, Iterator, Optional, List, Callable, Union

import torch
from torch import Tensor


def fit_readout(data: Iterator[Tuple[Tensor, Tensor]], regularization: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    """
    Ridge regression for big data
    Fits a linear model :math:`y = W x + b` with regularization.
    See:
    T. Zhang & B. Yang (2017). An exact approach to ridge regression for big data.
    Computational Statistics, 32(3), 909–928. https://doi.org/10.1007/s00180-017-0731-5
    :param data: Batch dataset of pairs (x, y) with samples on rows
    :param regularization: Regularization constant for ridge regression (default null)
    :return: A pair of tensors (W, b)
    """
    # Compute sufficient statistics for regression
    x, y = next(data)
    Syy = y.square().sum(dim=0)  # (targets)
    Sxy = x.t() @ y  # (features × targets)
    Sxx = x.t() @ x  # (features × features)
    Sy = y.sum(dim=0)  # (targets)
    Sx = x.sum(dim=0)  # (features)
    n = float(x.shape[0])  # samples
    for x, y in data:
        Syy += y.square().sum(dim=0)
        Sxy += x.t() @ y
        Sxx += x.t() @ x
        Sy += y.sum(dim=0)
        Sx += x.sum(dim=0)
        n += x.shape[0]
    # Compute ridge matrices
    Vxx = Sxx.diag() - (Sx.square() / n)
    Vyy = Syy - (Sy.square() / n)
    XX = (Sxx - torch.outer(Sx, Sx) / n) / torch.outer(Vxx, Vxx).sqrt()
    Xy = (Sxy - torch.outer(Sx, Sy) / n) / torch.outer(Vxx, Vyy).sqrt()
    if regularization:
        XX += torch.eye(n=XX.shape[0]).to(XX) * regularization
    # Compute weights
    Ws = torch.linalg.solve(XX, Xy)
    W = Ws * torch.sqrt(Vyy.expand_as(Ws) / Vxx.unsqueeze(-1))
    b = (Sy / n) - (Sx / n) @ W
    return W.t(), b


def fit_and_validate_readout(data: Iterator[Tuple[Tensor, Tensor]], regularization_constants: List[float],
                             get_validation_error: Callable[[Tuple[Tensor, Tensor]], float],
                             verbose: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Ridge regression for big data, with efficient regularization selection
    Fits a linear model :math:`y = W x + b` with regularization.
    See:
    T. Zhang & B. Yang (2017). An exact approach to ridge regression for big data.
    Computational Statistics, 32(3), 909–928. https://doi.org/10.1007/s00180-017-0731-5
    :param data: Batch dataset of pairs (x, y) with samples on rows
    :param regularization_constants: Regularization constants for ridge regression (including none)
    :param get_validation_error: Evaluate validation error for a regression pair (W, b)
    :param verbose: Whether to print validation info (default false)
    :return: A pair of tensors (W, b)
    """
    # Compute sufficient statistics for regression
    x, y = next(data)
    Syy = y.square().sum(dim=0)  # (targets)
    Sxy = x.t() @ y  # (features × targets)
    Sxx = x.t() @ x  # (features × features)
    Sy = y.sum(dim=0)  # (targets)
    Sx = x.sum(dim=0)  # (features)
    n = float(x.shape[0])  # samples
    for x, y in data:
        Syy += y.square().sum(dim=0)
        Sxy += x.t() @ y
        Sxx += x.t() @ x
        Sy += y.sum(dim=0)
        Sx += x.sum(dim=0)
        n += x.shape[0]
    # Compute ridge matrices
    Vxx = Sxx.diag() - (Sx.square() / n)
    Vyy = Syy - (Sy.square() / n)
    XX = (Sxx - torch.outer(Sx, Sx) / n) / torch.outer(Vxx, Vxx).sqrt()
    Xy = (Sxy - torch.outer(Sx, Sy) / n) / torch.outer(Vxx, Vyy).sqrt()
    # Compute and select weights
    best_validation_error, best_W, best_b = None, None, None
    for regularization in regularization_constants:
        # Compute weights
        XXr = (XX + torch.eye(n=XX.shape[0]).to(XX) * regularization) if regularization else XX
        Ws = torch.linalg.solve(XXr, Xy)
        W = Ws * torch.sqrt(Vyy.expand_as(Ws) / Vxx.unsqueeze(-1))
        b = (Sy / n) - (Sx / n) @ W
        # Validate, select
        validation_error = get_validation_error((W.t(), b))
        if best_validation_error is None or validation_error < best_validation_error:
            best_validation_error, best_W, best_b = validation_error, W.t(), b
        if verbose:
            print(f'{regularization:e}: {validation_error}', file=sys.stderr)
    return best_W, best_b