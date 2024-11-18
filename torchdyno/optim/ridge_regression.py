from operator import itemgetter
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


@torch.no_grad()
def fit_and_validate_readout(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    l2_values: List[float],
    score_fn: Callable[[Tensor, Tensor], float],
    mode: Literal["min", "max"],
    weights: Optional[List[float]] = None,
    preprocess_fn: Optional[Callable] = None,
    skip_first_n: int = 0,
    device: Optional[str] = None,
) -> Tuple[Tensor, float, float, Tensor, Tensor]:
    """Applies the ridge regression on the training data with all the given l2 values,
    and returns the best configuration after evaluating the linear transformations on
    the validation data.

    Args:
        train_loader (DataLoader): DataLoader of the training data.
        eval_loader (DataLoader): DataLoader of the validation data.
        l2_values (List[float]): List of all the candidate L2 values.
        score_fn (Callable[[Tensor, Tensor], float]): a Callable which, if applied to
            the predicted `y_pred` and the ground-truth `y_true`, returns the desired
            metric.
        mode (Literal[&#39;min&#39;, &#39;max&#39;]): whether the best result is the
            minimum or the maximum given the metric.
        weights (Optional[List[float]], optional): list of weights to be applied to each
            sample in the batch. Defaults to None.
        preprocess_fn (Optional[Callable], optional): a transformation to be applied to
            X before the linear transformation. Useful whenever this function is called
            to learn a Readout of a ESN. Defaults to None.
        skip_first_n (Optional[int], optional): number of samples to skip in each batch
            of the train_loader. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed.
            If None, the function is executed on a CUDA device if available, on CPU
            otherwise. Defaults to None.

    Returns:
        Tuple[Tensor, float, float, Tensor, Tensor]: a Tuple containing the best linear
            transformation, the corrisponding l2 value, metric value, ridge matrice B and
            ridge matrix B.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    all_W, A, B = fit_readout(
        train_loader=train_loader,
        preprocess_fn=preprocess_fn,
        l2=l2_values,
        weights=weights,
        skip_first_n=skip_first_n,
        device=device,
    )
    if not isinstance(all_W, list):
        all_W = [all_W]

    # Validation
    eval_scores = validate_readout(
        readout=all_W,
        eval_loader=eval_loader,
        score_fn=score_fn,
        preprocess_fn=preprocess_fn,
        skip_first_n=skip_first_n,
        device=device,
    )

    if not isinstance(eval_scores, list):
        return all_W[0], l2_values[0], eval_scores, A, B

    # Selection
    select_fn = max if mode == "max" else min
    best_idx, best_score = select_fn(enumerate(eval_scores), key=itemgetter(1))
    return all_W[best_idx], l2_values[best_idx], best_score, A, B


@torch.no_grad()
def fit_readout(
    train_loader: DataLoader,
    preprocess_fn: Optional[Callable] = None,
    l2: Optional[Union[float, List[float]]] = None,
    weights: Optional[List[float]] = None,
    skip_first_n: int = 0,
    device: Optional[str] = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Applies the ridge regression on the training data with all the given l2 values
    and returns a list of matrices, one for each L2 value.

    Args:
        train_loader (DataLoader): DataLoader of the training data.
        preprocess_fn (Optional[Callable], optional): a transformation to be applied to
            X before the linear transformation. Useful whenever this function is called
            to learn a Readout of a ESN. Defaults to None.
        l2_values (List[float]): List of all the candidate L2 values.
        weights (Optional[List[float]], optional): list of weights to be applied to each
            sample in the batch. Defaults to None.
        skip_first_n (Optional[int], optional): number of samples to skip in each batch
            of the train_loader. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed.
            If None, the function is executed on a CUDA device if available, on CPU
            otherwise. Defaults to None.

    Returns:
        Tuple[Tensor, float, float]: a Tuple containing the best linear matrix, the
            corrisponding l2 value and the metric value.
    """
    A, B = compute_ridge_matrices(
        loader=train_loader,
        preprocess_fn=preprocess_fn,
        weights=weights,
        skip_first_n=skip_first_n,
        device=device,
    )
    if isinstance(l2, List):
        readout = [
            solve_ab_decomposition(A=A, B=B, l2=curr_l2, device=device)
            for curr_l2 in l2
        ]
    else:
        readout = solve_ab_decomposition(A=A, B=B, l2=l2, device=device)

    return readout, A, B


@torch.no_grad()
def validate_readout(
    readout: Union[torch.Tensor, List[torch.Tensor]],
    eval_loader: DataLoader,
    score_fn: Callable[[Tensor, Tensor], float],
    preprocess_fn: Optional[Callable] = None,
    skip_first_n: int = 0,
    device: Optional[str] = None,
):
    """Evaluates the linear transformations on the validation data.

    Args:
        readout (Union[torch.Tensor, List[torch.Tensor]]): list of readouts to validate.
        eval_loader (DataLoader): DataLoader of the validation data.
        score_fn (Callable[[Tensor, Tensor], float]): a Callable which,
            if applied to the predicted `y_pred` and the ground-truth `y_true`,
            returns the desired metric.
        preprocess_fn (Optional[Callable], optional): a transformation to be applied
            to X before the linear transformation. Useful whenever this function is
            called to learn a Readout of a ESN. Defaults to None.
        skip_first_n (Optional[int], optional): number of samples to skip in each batch
            of the train_loader. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed.
            If None, the function is executed on a CUDA device if available, on CPU
            otherwise. Defaults to None.

    Returns:
        List[float]: a list containing the metric values.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(readout, list):
        readout = [readout]

    # Validation
    all_W = [w.to(device) for w in readout]
    eval_scores, n_samples = [0 for _ in range(len(readout))], 0
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        # Processing x
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        size_x = x.size()
        size_y = y.size()
        if len(size_x) > 2:
            x = x.reshape(-1, size_x[-1])
            y = y.reshape(-1, size_y[-1])

        x, y = x[skip_first_n:], y[skip_first_n:]
        curr_n_samples = x.size(0)
        # Computing scores
        for i, W in enumerate(all_W):
            y_pred = F.linear(x.to(W), W)
            score_W = score_fn(y, y_pred)
            eval_scores[i] += score_W * curr_n_samples

        n_samples += curr_n_samples

    results = [score / n_samples for score in eval_scores]
    return results if len(results) > 1 else results[0]


@torch.no_grad()
def compute_ridge_matrices(
    loader: DataLoader,
    preprocess_fn: Optional[Callable] = None,
    weights: Optional[List[float]] = None,
    skip_first_n: int = 0,
    device: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Computes the matrices A and B for incremental ridge regression. For each batch in
    the loader, it applies the preprocess_fn on  the x sample, resizes it to (n_samples,
    hidden_size), and computes the values of A and B.

    Args:
        loader (DataLoader): torch loader
        preprocess_fn (Optional[Callable], optional): function to be applied to the x sample
            before computing the matrices. Defaults to None.
        weights (Optional[List[float]], optional): list of weights to be applied to each
            sample in the batch. Defaults to None.
        skip_first_n (Optional[int], optional): number of samples to skip in each batch
            of the train_loader. Defaults to None.
        device (Optional[str], optional): the device on which the function is executed.
            If None, the function is executed on a CUDA device if available, on CPU
            otherwise. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: the matrices A of shape
            [label_size x hidden_size] and B of shape
            [hidden_size x hidden_size].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if weights is not None:
        weights = torch.tensor(weights).to(device)

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

        x, y = x[skip_first_n:], y[skip_first_n:]

        batch_A, batch_B = (y.T @ x).cpu(), (x.T @ x).cpu()

        if weights is not None:
            curr_w = weights[y.long()[:, 0]]
            batch_A, batch_B = ((y.T * curr_w) @ x).cpu(), ((x.T * curr_w) @ x).cpu()
        else:
            batch_A, batch_B = (y.T @ x).cpu(), (x.T @ x).cpu()

        A, B = (A + batch_A, B + batch_B) if A is not None else (batch_A, batch_B)

    return A, B


@torch.no_grad()
def solve_ab_decomposition(
    A: Tensor, B: Tensor, l2: Optional[float] = None, device: Optional[str] = None
) -> Tensor:
    """Computes the result of the AB decomposition for solving the linear system.

    Args:
        A (Tensor): YS^T, where Y is the target matrix and S is the input matrix.
        B (Tensor): SS^T, where S is the input matrix.
        l2 (Optional[float], optional): the value of l2 regularization. Defaults to None.

    Returns:
        Tensor: matrix W of shape [label_size x hidden_size]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    A, B = A.to(device), B.to(device)
    B = B + torch.eye(B.shape[0]).to(B) * l2 if l2 else B

    return A @ B.pinverse()


@torch.no_grad()
def compress_ridge_matrices(
    A: Tensor, B: Tensor, perc_rec: float, alpha: float
) -> Tuple[Tensor, Tensor]:
    """Masks the matrices A and B according to the percentage of recurrent neurons to be
    used. The `perc_rec` percentage of the most important recurrent neurons are used,
    where the importance is measured by the sum of the squares of the columns of B.

    Args:
        A (Tensor): YS^T
        B (Tensor): SS^T
        perc_rec (Optional[float], optional): percentage of the recurrent neurons to be used.
            If None, all the recurrent neurons are used. Defaults to None.
        alpha (Optional[float], 1.0): use alpha recurrent neurons based on importance and (1-alpha)
            random neurons over the fraction of all recurrent neurons given by `perc_rec`.
            Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor]: the masked matrices A and B.

    Raises:
        ValueError: if perc_rec or alpha are not in [0, 1]
    """
    if perc_rec < 0 or perc_rec > 1:
        raise ValueError("perc_rec must be in [0, 1]")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")

    # number of recurrent neurons to be considered
    n = int(perc_rec * B.size(0))
    # fraction of top-k and random neurons
    all_idxs = list(range(B.size(0)))
    k, k_rand = int(round(alpha * n)), int((1 - alpha) * n)

    if alpha > 0:
        # compute the importance of each column of B
        imp = torch.sum(B**2, axis=1)
        _, topk_idxs = torch.topk(imp, k)
    else:
        topk_idxs = torch.tensor([])

    if alpha < 1:
        rand_idxs = torch.tensor(list(set(all_idxs) - set(topk_idxs.tolist())))
        randperm_idxs = torch.randperm(len(rand_idxs))
        rand_idxs = rand_idxs[randperm_idxs][:k_rand]
    else:
        rand_idxs = torch.tensor([])

    chosen_idxs = torch.hstack((topk_idxs, rand_idxs)).long()
    mask = F.one_hot(chosen_idxs, B.size(0)).sum(0).unsqueeze(0)

    masked_A = A * mask
    masked_B = (mask.T @ mask) * B

    return masked_A, masked_B
