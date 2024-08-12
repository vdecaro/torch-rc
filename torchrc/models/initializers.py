from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import scipy.sparse
import numpy as np


def get_initializer(
    name: str,
) -> Callable:
    """
    Get initializer function by name.

    Args:
        name (str): name of the initializer.
    Returns:
        Callable: initializer function.
    """

    if name == "uniform":
        return uniform
    elif name == "normal":
        return normal
    elif name == "ring":
        return ring
    elif name == "orthogonal":
        return orthogonal
    elif name == "ones":
        return ones
    elif name == "zeros":
        return zeros
    elif name == "block_diagonal":
        return block_diagonal
    else:
        raise ValueError(f"Unknown initializer: {name}")


def rescale(
    tensor: torch.Tensor,
    method: Literal["spectral", "singular", "norm", "linear"],
    value: float,
) -> torch.Tensor:
    """
    Rescale a matrix in-place. Can either be rescaled according to spectral radius
    `rho`, spectral norm `sigma`, or `scale`.

    Args:
        shape (torch.Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        torch.Tensor: initialized tensor.
    """
    if method == "spectral":
        return tensor.div_(torch.linalg.eigvals(tensor).abs().max()).mul_(value).float()
    elif method == "singular":
        return tensor.div_(torch.linalg.svdvals(tensor).abs().max()).mul_(value).float()
    elif method == "norm":
        return tensor.div_(torch.linalg.matrix_norm(tensor, ord=2)).mul_(value).float()
    elif method == "linear":
        return tensor.mul_(value).float()


def uniform(
    shape: torch.Size,
    min_val: float = -1,
    max_val: float = 1,
) -> torch.Tensor:
    """
    Uniform random tensor.

    Args:
        shape (torch.Size): shape of the tensor.
        min_val (float, optional): minimum value for uniform initialization. Defaults to -1.
        max_val (float, optional): maximum value for uniform initialization. Defaults to 1.

    Returns:
        torch.Tensor: initialized tensor.
    """
    return torch.empty(shape).uniform_(min_val, max_val)


def normal(
    shape: torch.Size,
    mean: float = 0,
    std: float = 1,
) -> torch.Tensor:
    """
    Normal random tensor. Can either be rescaled according to spectral radius `rho`,
    spectral norm `sigma`, or `scale`.

    Args:
        shape (torch.Size): shape of the tensor.
        mean (float, optional): mean value for normal initialization. Defaults to 0.
        std (float, optional): standard deviation for normal initialization. Defaults to 1.

    Returns:
        torch.Tensor: initialized tensor.
    """
    return torch.empty(shape).normal_(mean=mean, std=std)


def ring(
    n_units: int,
    values: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Ring matrix. See:
        C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
        In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
        https://doi.org/10.1109/IJCNN48605.2020.9206723

    Args:
        shape (torch.Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        torch.Tensor: initialized ring tensor.
    """
    tensor = torch.eye(n_units).roll(1, 0)

    if values is None:
        values = uniform((n_units,))
    tensor.mul_(values)
    return tensor


def orthogonal(
    shape: torch.Size,
    gain: float = 1,
) -> torch.Tensor:
    """
    Orthogonal matrix. See:
        F. Mezzadri (2007). How to Generate Random Matrices from the Classical Compact
        Groups. Notices of the American Mathematical Society, 54(5), pp. 592-604.
        https://www.ams.org/notices/200705/fea-mezzadri-web.pdf

    Args:
        shape (torch.Size): shape of the tensor.
        gain (float, optional): scaling value. Defaults to 1.

    Returns:
        torch.Tensor: initialized orthogonal tensor.
    """

    mat = torch.empty(shape)
    torch.nn.init.orthogonal_(mat, gain)
    return mat


def ones(shape: torch.Size) -> torch.Tensor:
    """
    Ones tensor.

    Args:
        shape (torch.Size): shape of the tensor.

    Returns:
        torch.Tensor: initialized ones tensor.
    """

    return torch.ones(shape)


def zeros(shape: torch.Size) -> torch.Tensor:
    """
    Zeros tensor.

    Args:
        shape (torch.Size): shape of the tensor

    Returns:
        torch.Tensor: zeros tensor.
    """
    return torch.zeros(shape)


def sparse(
    shape: torch.Size,
    density: float = 0.01,
    values_sampler: Optional[Callable[..., np.ndarray]] = None,
    enforce_cond: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sparse random tensor.

    Args:
        shape (torch.Size): shape of the tensor.

    Returns:
        torch.Tensor: initialized sparse random tensor.
    """
    # use scipy.sparse.random to generate sparse random matrix
    if values_sampler is None:
        values_sampler = lambda x: np.random.uniform(low=-1.0, high=1.0, size=x)
    while True:
        sparse_mat = scipy.sparse.random(
            shape[0],
            shape[1],
            density=density,
            data_rvs=values_sampler,
            random_state=seed,
        ).toarray()
        np.fill_diagonal(sparse_mat, 0)
        if not enforce_cond or _check_cond(sparse_mat):
            break
    return torch.tensor(sparse_mat).float()


def block_diagonal(
    blocks: List[torch.Tensor],
    couplings: Optional[List[Tuple[int, int, torch.Tensor]]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a block diagonal matrix from a list of matrices.

    Args:
        blocks (torch.Tensor): list of matrices.
        couplings (Optional[List[Tuple[int, int, torch.Tensor]]], optional): list of
            blocks coupling blocks in the diagonal. Defaults to None.

    Returns:
        torch.Tensor: block diagonal matrix.
    """
    n_total = sum([b.shape[0] for b in blocks])

    mat = torch.zeros(n_total, n_total)
    n_units = blocks[0].shape[0]
    for i, block in enumerate(blocks):
        curr_units = block.shape[0]
        extent = n_units + curr_units
        mat[n_units:extent, n_units:extent] = block

    if couplings is not None:
        margin_x, margin_y = [0], [blocks[0].shape[0]]
        for i in range(1, len(blocks)):
            margin_x.append(margin_x[-1] + blocks[i - 1].shape[0])
            margin_y.append(margin_y[-1] + blocks[i].shape[0])
        couple_mat = torch.zeros(n_total, n_total)
        for i, j, corr in couplings:
            if i == j:
                raise ValueError(
                    f"Coupling blocks must couple different diagonal blocks. Found repeated {i}."
                )

            if i > j:
                i, j = j, i
                corr = corr.T

            couple_mat[margin_x[i] : margin_x[i + 1], margin_y[j - 1] : margin_y[j]] = (
                corr
            )
        return mat, couple_mat


def _check_cond(W):
    W_diag_only = np.diag(np.diag(W))
    W_diag_pos_only = W_diag_only.copy()
    W_diag_pos_only[W_diag_pos_only < 0] = 0.0
    W_abs_cond = np.abs(W - W_diag_only) + W_diag_pos_only
    max_eig_abs_cond = np.max(np.real(np.linalg.eigvals(W_abs_cond)))
    return max_eig_abs_cond < 1


__all__ = ["uniform", "normal", "ring", "orthogonal", "ones", "zeros", "block_diagonal"]
