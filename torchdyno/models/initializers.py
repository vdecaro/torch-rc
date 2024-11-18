from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import scipy.sparse
import torch


def rescale(
    tensor: torch.Tensor,
    method: Literal["spectral", "singular", "norm", "linear"] = "spectral",
    value: float = 0.999,
) -> torch.Tensor:
    """Rescale a matrix in-place. Rescaling can be done according to the spectral
    radius, spectral norm, matrix norm, or linear scaling.

    Args:
        shape (torch.Size): shape of the tensor.
        method (Literal["spectral", "singular", "norm", "linear"]): rescaling method.
        value (float): scaling value.

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
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform random tensor.

    Args:
        shape (torch.Size): shape of the tensor.
        min_val (float, optional): minimum value for uniform initialization. Defaults to -1.
        max_val (float, optional): maximum value for uniform initialization. Defaults to 1.

    Returns:
        torch.Tensor: initialized tensor.
    """
    if min_val is None:
        min_val = -1
    if max_val is None:
        max_val = -min_val

    return torch.empty(shape, dtype=dtype).uniform_(min_val, max_val)


def normal(
    shape: torch.Size,
    mean: float = 0,
    std: float = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normal random tensor. Can either be rescaled according to spectral radius `rho`,
    spectral norm `sigma`, or `scale`.

    Args:
        shape (torch.Size): shape of the tensor.
        mean (float, optional): mean value for normal initialization. Defaults to 0.
        std (float, optional): standard deviation for normal initialization. Defaults to 1.

    Returns:
        torch.Tensor: initialized tensor.
    """
    return torch.empty(shape, dtype=dtype).normal_(mean=mean, std=std)


def ring(
    n_units: int,
    values: Optional[Union[float, torch.Tensor]] = None,
    dtype: torch.dtype = torch.float32,
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
    tensor = torch.eye(n_units, dtype=dtype).roll(1, 0)

    if values is None:
        values = uniform((n_units,), dtype=dtype)
    tensor.mul_(values)
    return tensor


def lower_feedforward(
    shape: torch.Size, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Lower feedforward matrix.

    Args:
        shape (torch.Size): shape of the tensor.
        dtype (torch.dtype, optional): data type. Defaults to torch.float32.

    Returns:
        torch.Tensor: initialized lower feedforward tensor.
    """
    if len(shape) != 2:
        raise ValueError("Lower feedforward matrix must be 2D.")
    if shape[0] != shape[1]:
        raise ValueError("Lower feedforward matrix must be square.")

    tensor = diagonal(shape[0], dtype=dtype)
    idx = round(shape[0] / 2)
    tensor[idx:, :idx] = uniform((shape[0] - idx, shape[0] - idx), dtype=dtype)
    return tensor


def orthogonal(shape: torch.Size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
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

    mat = torch.empty(shape, dtype=dtype)
    torch.nn.init.orthogonal_(mat)
    return mat


def ones(shape: torch.Size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ones tensor.

    Args:
        shape (torch.Size): shape of the tensor.

    Returns:
        torch.Tensor: initialized ones tensor.
    """

    return torch.ones(shape, dtype=dtype)


def zeros(shape: torch.Size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Zeros tensor.

    Args:
        shape (torch.Size): shape of the tensor

    Returns:
        torch.Tensor: zeros tensor.
    """
    return torch.zeros(shape, dtype=dtype)


def diagonal(
    shape: torch.Size,
    min_val: float = -1,
    max_val: float = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Diagonal random tensor.

    Args:
        shape (int): shape of the tensor.
        min_val (float, optional): minimum value for uniform initialization. Defaults to -1.
        max_val (float, optional): maximum value for uniform initialization. Defaults to 1.

    Returns:
        torch.Tensor: initialized tensor.
    """
    return torch.diag(torch.empty(shape[0], dtype=dtype).uniform_(min_val, max_val))


def sparse(
    shape: torch.Size,
    density: float = 0.01,
    values_sampler: Optional[Callable[..., np.ndarray]] = None,
    zero_diagonal: bool = True,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sparse random tensor.

    Args:
        shape (torch.Size): shape of the tensor.
        density (float, optional): density of the sparse tensor. Defaults to 0.01.
        values_sampler (Optional[Callable[..., np.ndarray]], optional): function to sample
            values for the sparse tensor. Defaults to None.
        zero_diagonal (bool, optional): whether to zero the diagonal. Defaults to True.
        seed (Optional[int], optional): random seed. Defaults to None.

    Returns:
        torch.Tensor: initialized sparse random tensor.
    """
    # use scipy.sparse.random to generate sparse random matrix
    npdtype = _torch_to_numpy_dtype(dtype)
    if values_sampler is None:
        values_sampler = lambda x: np.random.uniform(
            low=-1.0, high=1.0, size=x, dtype=npdtype
        )
    sparse_mat = scipy.sparse.random(
        shape[0],
        shape[1],
        density=density,
        data_rvs=values_sampler,
        random_state=seed,
        dtype=npdtype,
    ).toarray()
    if zero_diagonal:
        np.fill_diagonal(sparse_mat, 0)
    return torch.tensor(sparse_mat).float()


def block_diagonal(
    blocks: List[torch.Tensor],
) -> torch.Tensor:
    """Create a block diagonal matrix from a list of matrices.

    Args:
        blocks (torch.Tensor): list of matrices.
        couplings (Optional[List[Tuple[int, int, torch.Tensor]]], optional): list of
            blocks coupling blocks in the diagonal. Defaults to None.

    Returns:
        torch.Tensor: block diagonal matrix.
    """
    n_total = sum([b.shape[0] for b in blocks])

    mat = torch.zeros(n_total, n_total)
    curr_idx = 0
    for block in blocks:
        curr_units = block.shape[0]
        extent = curr_idx + curr_units
        mat[curr_idx:extent, curr_idx:extent] = block
        curr_idx = extent
    return mat


def block_diagonal_coupling(
    block_sizes: List[int], couplings: List[Tuple[int, int, torch.Tensor]]
) -> torch.Tensor:
    """Create the coupling matrix for a given block diagonal matrix.

    Args:
        block_sizes (List[int]): list of block sizes.
        couplings (List[Tuple[int, int, torch.Tensor]]): list of coupling blocks.

    Returns:
        torch.Tensor: coupling matrix
    """

    n_total = sum(block_sizes)
    margin_x, margin_y = [0], [block_sizes[0]]
    for i in range(1, len(block_sizes)):
        margin_x.append(margin_x[-1] + block_sizes[i - 1])
        margin_y.append(margin_y[-1] + block_sizes[i])

    couple_mat = torch.zeros(n_total, n_total)
    for i, j, corr in couplings:
        if i == j:
            raise ValueError(
                f"Coupling blocks must couple different diagonal blocks. Found repeated {i}."
            )

        if i > j:
            i, j = j, i
            corr = corr.T

        couple_mat[margin_x[i] : margin_x[i + 1], margin_y[j - 1] : margin_y[j]] = corr
    return couple_mat


def _torch_to_numpy_dtype(torch_dtype):
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }
    return mapping.get(torch_dtype, None)


__all__ = [
    "rescale",
    "uniform",
    "normal",
    "ring",
    "lower_feedforward",
    "orthogonal",
    "ones",
    "zeros",
    "diagonal",
    "sparse",
    "block_diagonal",
    "block_diagonal_coupling",
]
