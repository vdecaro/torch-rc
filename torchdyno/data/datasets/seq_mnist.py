from typing import (
    Callable,
    Optional,
    Tuple,
)

import torch
from torchvision import transforms
from torchvision.datasets import MNIST


class SequentialMNIST(MNIST):
    """Sequential MNIST dataset.

    The Sequential MNIST dataset is a variant of the MNIST dataset where the pixels of
    the images are permuted in a fixed way. Each image is treated pixel by pixel as a
    sequence, resulting in the concatenation of the rows of the image.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable[..., torch.Tensor]] = None,
        target_transform: Optional[Callable[..., torch.Tensor]] = None,
        download: bool = False,
        permute_seed: Optional[int] = None,
    ):
        """Sequential MNIST dataset.

        Args:
            root (str, optional): root directory of dataset.
            train (bool, optional): whether to load the training or test set. Defaults to True.
            transform (Optional[Callable[..., torch.Tensor]], optional): a function/transform that takes in an PIL image and returns a transformed version. Defaults to None.
            target_transform (Optional[Callable[..., torch.Tensor]], optional): a function/transform that takes in the target and transforms it. Defaults to None.
            download (bool, optional): whether to download the dataset. Defaults to False.
            permute_seed (Optional[int], optional): seed for permutation. Defaults to None.
        """
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, train, transform, target_transform, download)
        self.permute_seed = permute_seed

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the item at the given index."""
        raw = super().__getitem__(index)
        img: torch.Tensor = raw[0]
        target: torch.Tensor = raw[1]
        if self.permute_seed is not None:
            img = img.view(-1)[
                torch.randperm(
                    img.numel(),
                    generator=torch.Generator().manual_seed(self.permute_seed),
                )
            ].view(img.size())

        img = img.view(-1).unsqueeze(-1)
        return img, target
