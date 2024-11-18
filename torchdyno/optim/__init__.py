"""Package for optimization functionalities."""

from . import ridge_regression as rr
from .intrinsic_plasticity import IntrinsicPlasticity

__all__ = ["rr", "IntrinsicPlasticity"]
