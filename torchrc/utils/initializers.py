from typing import Callable, List, Literal, Optional, Tuple, Union

import torch


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


def block_diagonal(
    blocks: List[torch.Tensor],
    correlations: Optional[List[Tuple[int, int, torch.Tensor]]] = None,
    antisymmetric_corrs: bool = False,
    decompose: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a block diagonal matrix from a list of matrices.

    Args:
        blocks (torch.Tensor): list of matrices.
        correlations (Optional[List[Tuple[int, int, torch.Tensor]]], optional): list of
            blocks correlating blocks in the diagonal. Defaults to None.
        antisymmetric_corrs (bool, optional): whether to use antisymmetric correlations.

    Returns:
        torch.Tensor: block diagonal matrix.
    """
    n_blocks = len(blocks)
    n_units = blocks[0].shape[0]
    n_total = n_units * n_blocks

    def _check_block(block: torch.Tensor, is_corr: bool = False):
        to_print = "Correlation matrix" if is_corr else "Block matrix"
        if block.ndim != 2:
            raise ValueError("Block is not a matrix.")
        if block.shape[0] != block.shape[1]:
            raise ValueError("Block is not square.")
        if block.shape[0] != n_units:
            raise ValueError(
                f"{to_print} has the wrong size. Expected {n_units}, got {block.shape[0]}."
            )

    for i, block in enumerate(blocks):
        _check_block(block)

    mat = torch.zeros(n_total, n_total)
    for i, block in enumerate(blocks):
        mat[i * n_units : (i + 1) * n_units, i * n_units : (i + 1) * n_units] = block

    if correlations is not None:
        if not decompose:
            for i, j, corr in correlations:
                _check_block(corr)

                mat[
                    i * n_units : (i + 1) * n_units, j * n_units : (j + 1) * n_units
                ] = corr
                mat[
                    j * n_units : (j + 1) * n_units, i * n_units : (i + 1) * n_units
                ] = (corr.T if not antisymmetric_corrs else -corr.T)
            return mat
        else:
            corr_mat = torch.zeros(n_total, n_total)
            for i, j, corr in correlations:
                _check_block(corr)

                corr_mat[
                    i * n_units : (i + 1) * n_units, j * n_units : (j + 1) * n_units
                ] = corr
                corr_mat[
                    j * n_units : (j + 1) * n_units, i * n_units : (i + 1) * n_units
                ] = (corr.T if not antisymmetric_corrs else -corr.T)
            return mat, corr_mat


__all__ = ["uniform", "normal", "ring", "orthogonal", "ones", "zeros", "block_diagonal"]
