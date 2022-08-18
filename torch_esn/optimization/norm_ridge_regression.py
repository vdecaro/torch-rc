import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch import Tensor
from typing import Literal, Optional, Tuple, List, Callable, Dict, Union
from operator import itemgetter


def fit_and_validate_readout(train_loader: DataLoader, 
                             eval_loader: DataLoader,
                             l2_values: List[float],
                             score_fn: Callable[[Tensor, Tensor], float],
                             mode: Literal['min', 'max'],
                             preprocess_fn: Optional[Callable] = None) -> Tuple[Tensor, Tensor]:
    # Training
    all_W = fit_readout(train_loader, preprocess_fn, l2_values)
    if not isinstance(all_W, List):
        all_W = [all_W]
    
    # Validation
    eval_scores, n_samples = [0 for _ in range(len(l2_values))], 0
    for x, y in eval_loader:

        # Processing x
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        size_x = x.size()
        if len(size_x) > 3:
            x = x.reshape(-1, size_x[-1])

        curr_n_samples = x.size(0)
        # Computing scores
        for i, W in enumerate(all_W):
            y_pred = F.linear(x.to(W), W)
            score_W = score_fn(y, y_pred)
            eval_scores[i] += score_W * curr_n_samples

        n_samples += curr_n_samples

    # Selection
    eval_scores = [score / n_samples for score in eval_scores]
    select_fn = max if mode == 'max' else min
    best_idx, best_score = select_fn(enumerate(eval_scores), key=itemgetter(1))

    return all_W[best_idx], l2_values[best_idx], best_score


def fit_readout(train_loader: torch.utils.data.DataLoader,
                preprocess_fn: Optional[Callable] = None,
                l2: Optional[Union[float, List[float]]] = None) -> Tuple[Tensor, Tensor]:

    A, B = compute_ridge_matrices(train_loader, preprocess_fn)
    if isinstance(l2, List):
        return [solve_ab_decomposition(A, B, curr_l2) for curr_l2 in l2]
    else:
        return solve_ab_decomposition(A, B, l2)


def compute_ridge_matrices(loader: DataLoader,
                           preprocess_fn: Optional[Callable] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the matrices A and B for incremental ridge regression. For each batch in the loader, it applies the
    preprocess_fn on the x sample, resizes it to (n_samples, hidden_size) and computes the values of A and B.

    Args:
        loader (DataLoader): torch loader
        preprocess_fn (Callable): function to be applied to the x sample

    Returns:
        Tuple[Tensor, Tensor]: the matrices A of shape [label_size x hidden_size] and B of shape [hidden_size x hidden_size]
    """
    A, B = None, None
    for x, y in loader:
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        size_x = x.size()
        size_y = y.size()
        if len(size_x) > 3:
            x = x.reshape(-1, size_x[-1])
            y = y.reshape(-1, size_y[-1])
        
        batch_A, batch_B = y.Y @ x, x.T @ x
        A, B = (A + batch_A, B + batch_B) if A is not None else (batch_A, batch_B)
    
    return A, B


def solve_ab_decomposition(A: Tensor, B: Tensor, l2: Optional[float] = None) -> Tensor:
    """
    Computes the result of the AB decomposition for solving the linear system

    Args:
        A (Tensor): YS^T
        B (Tensor): SS^T
        l2 (Optional[float], optional): The value of l2 regularization. Defaults to None.

    Returns:
        Tensor: matrix W of shape [label_size x hidden_size]
    """
    B = B + torch.eye(B.shape[0]).to(B) * l2 if l2 else B
    return A @ B.to('cpu').pinverse().to(A)