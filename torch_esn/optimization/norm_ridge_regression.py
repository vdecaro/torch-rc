import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple, List, Callable, Dict


############ CAMBIARE IN LOADER CON INPUT TRANSFORM
def fit_readout(X: Tensor, Y: Tensor, l2: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    A, B = compute_ridge_matrices(X, Y)
    return solve_ab_decomposition(A, B, l2)


def fit_and_validate_readout(train_X: Tensor, 
                             train_Y: Tensor,
                             eval_X: Tensor,
                             eval_Y: Tensor,
                             l2_values: List[float],
                             score_fn: Callable[[Tensor, Tensor], Dict]) -> Tuple[Tensor, Tensor]:
    
    best_W, best_l2, best_eval_score = None, None, None
    A, B = compute_ridge_matrices(train_X, train_Y)
    for l2 in l2_values:
        W = solve_ab_decomposition(A, B, l2)
        Y_pred = F.linear(eval_X, W)
        score = score_fn(eval_Y, Y_pred)

        if best_W is None or score > best_eval_score:
            best_W, best_l2, best_eval_score = W, l2, score
    
    return best_W, best_l2, best_eval_score

def validate_readout(A: Tensor, 
                     B: Tensor,
                     eval_data,
                     l2_values: List[float],
                     score_fn: Callable[[Tensor, Tensor], Dict]) -> Tuple[Tensor, Tensor]:
    
    best_W, best_l2, best_eval_score = None, None, None
    for l2 in l2_values:
        W = solve_ab_decomposition(A, B, l2)
        acc, n_samples = 0, 0
        for x, y in eval_data:
            Y_pred = torch.argmax(F.linear(x.to(W), W), dim=-1).flatten().to('cpu')
            curr_acc = score_fn(torch.argmax(y, dim=-1).flatten(), Y_pred)
            curr_n_samples = Y_pred.size(0)
            acc += curr_acc*curr_n_samples
            n_samples += curr_n_samples
        acc = acc / n_samples

        if best_W is None or acc > best_eval_score:
            best_W, best_l2, best_eval_score = W, l2, acc
    
    return best_W, best_l2, best_eval_score


def compute_ridge_matrices(X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes the matrices A and B for incremental ridge regression

    Args:
        X (Tensor): the input data (reservoir states) of shape [n_samples x hidden_size]
        Y (Tensor): the labels of shape [n_samples x label_size]

    Returns:
        Tuple[Tensor, Tensor]: the matrices A of shape [label_size x hidden_size] and B of shape [hidden_size x hidden_size]
    """
    A = Y.T @ X
    B = X.T @ X
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