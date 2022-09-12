import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch import Tensor
from typing import Literal, Optional, Tuple, List, Callable, Union
from operator import itemgetter


@torch.no_grad()
def fit_and_validate_readout(train_loader: DataLoader, 
                             eval_loader: DataLoader,
                             l2_values: List[float],
                             score_fn: Callable[[Tensor, Tensor], float],
                             mode: Literal['min', 'max'],
                             preprocess_fn: Optional[Callable] = None,
                             device: Optional[str] = None) -> Tuple[Tensor, Tensor]:
    """Applies the ridge regression on the training data with all the given l2 values
    and returns the best configuration after evaluating the linear transformations on the
    validation data.

    Args:
        train_loader (DataLoader): DataLoader of the training data.
        eval_loader (DataLoader): DataLoader of the validation data.
        l2_values (List[float]): List of all the candidate L2 values.
        score_fn (Callable[[Tensor, Tensor], float]): a Callable which, 
            if applied to the predicted `y_pred` and the ground-truth `y_true`, returns the desired metric.
        mode (Literal[&#39;min&#39;, &#39;max&#39;]): whether the best result is the minimum or the maximum given the metric.
        preprocess_fn (Optional[Callable], optional): a transformation to be applied to X before the linear transformation.
            Useful whenever this function is called to learn a Readout of a ESN. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed. If None, the function is executed on
            a CUDA device if available, on CPU otherwise. Defaults to None.

    Returns:
        Tuple[Tensor, float, float]: a Tuple containing the best linear matrix, the corrisponding l2 value and the metric value.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training
    all_W = fit_readout(train_loader, preprocess_fn, l2_values, device)
    if not isinstance(all_W, List):
        all_W = [all_W]
    
    # Validation
    eval_scores = validate_readout(all_W, eval_loader, score_fn, preprocess_fn, device)

    # Selection
    select_fn = max if mode == 'max' else min
    best_idx, best_score = select_fn(enumerate(eval_scores), key=itemgetter(1))

    return all_W[best_idx], l2_values[best_idx], best_score

@torch.no_grad()
def fit_readout(train_loader: torch.utils.data.DataLoader,
                preprocess_fn: Optional[Callable] = None,
                l2: Optional[Union[float, List[float]]] = None,
                device: Optional[str] = 'cpu') -> Tuple[Tensor, Tensor]:
    """Applies the ridge regression on the training data with all the given l2 values
    and returns a list of matrices, one for each L2 value.

    Args:
        train_loader (DataLoader): DataLoader of the training data.
        l2_values (List[float]): List of all the candidate L2 values.
        preprocess_fn (Optional[Callable], optional): a transformation to be applied to X before the linear transformation.
            Useful whenever this function is called to learn a Readout of a ESN. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed. If None, the function is executed on
            a CUDA device if available, on CPU otherwise. Defaults to None.

    Returns:
        Tuple[Tensor, float, float]: a Tuple containing the best linear matrix, the corrisponding l2 value and the metric value.
    """
    A, B = compute_ridge_matrices(train_loader, preprocess_fn, device)
    if isinstance(l2, List):
        return [solve_ab_decomposition(A, B, curr_l2, device) for curr_l2 in l2]
    else:
        return solve_ab_decomposition(A, B, l2, device)

@torch.no_grad()
def validate_readout(readout: Union[torch.Tensor, List[torch.Tensor]],
                     eval_loader: torch.utils.data.DataLoader,
                     score_fn: Callable[[Tensor, Tensor], float],
                     preprocess_fn: Optional[Callable] = None,
                     device: Optional[str] = 'cpu'):
    """Evaluates the linear transformations on the validation data.

    Args:
        readout (Union[torch.Tensor, List[torch.Tensor]]): list of readouts to validate.
        eval_loader (DataLoader): DataLoader of the validation data.
        score_fn (Callable[[Tensor, Tensor], float]): a Callable which, 
            if applied to the predicted `y_pred` and the ground-truth `y_true`, returns the desired metric.
        preprocess_fn (Optional[Callable], optional): a transformation to be applied to X before the linear transformation.
            Useful whenever this function is called to learn a Readout of a ESN. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed. If None, the function is executed on
            a CUDA device if available, on CPU otherwise. Defaults to None.

    Returns:
        List[float]: a list containing the metric values.
    """
    if not isinstance(readout, List):
        readout = [readout]
    
    # Validation
    all_W = [w.to(device) for w in readout]
    eval_scores, n_samples = [0 for _ in range(len(readout))], 0
    for x, y in eval_loader:
        x = x.to(device)
        # Processing x
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        size_x = x.size()
        size_y = y.size()
        if len(size_x) > 2:
            x = x.reshape(-1, size_x[-1])
            y = y.reshape(-1, size_y[-1])

        curr_n_samples = x.size(0)
        # Computing scores
        for i, W in enumerate(all_W):
            y_pred = F.linear(x.to(W), W)
            score_W = score_fn(y, y_pred)
            eval_scores[i] += score_W * curr_n_samples

        n_samples += curr_n_samples
    
    return [score / n_samples for score in eval_scores]

@torch.no_grad()
def compute_ridge_matrices(loader: DataLoader,
                           preprocess_fn: Optional[Callable] = None,
                           device: Optional[str] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the matrices A and B for incremental ridge regression. For each batch in the loader, it applies the
    preprocess_fn on the x sample, resizes it to (n_samples, hidden_size) and computes the values of A and B.

    Args:
        loader (DataLoader): torch loader
        preprocess_fn (Callable): function to be applied to the x sample

    Returns:
        Tuple[Tensor, Tensor]: the matrices A of shape [label_size x hidden_size] and B of shape [hidden_size x hidden_size]
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    A, B = None, None
    for x, y in loader:
        x = x.to(device)
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        size_x = x.size()
        size_y = y.size()
        if len(size_x) > 2:
            x = x.reshape(-1, size_x[-1])
            y = y.reshape(-1, size_y[-1])
        
        y = y.to(device).float()
        batch_A, batch_B = (y.T @ x).cpu(), (x.T @ x).cpu()
        A, B = (A + batch_A, B + batch_B) if A is not None else (batch_A, batch_B)
    
    return A, B

@torch.no_grad()
def solve_ab_decomposition(A: Tensor, B: Tensor, l2: Optional[float] = None, device: Optional[str] = 'cpu') -> Tensor:
    """
    Computes the result of the AB decomposition for solving the linear system

    Args:
        A (Tensor): YS^T
        B (Tensor): SS^T
        l2 (Optional[float], optional): The value of l2 regularization. Defaults to None.

    Returns:
        Tensor: matrix W of shape [label_size x hidden_size]
    """
    A = A.to(device)
    B = B + torch.eye(B.shape[0]).to(B) * l2 if l2 else B
    return A @ B.to('cpu').pinverse().to(A)