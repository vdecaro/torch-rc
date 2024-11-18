from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import numpy as np
import torch
from torch.utils.data import Dataset


class LorenzSystem(Dataset):
    """Lorenz system dataset.

    The Lorenz system is a system of ordinary differential equations that exhibit chaotic behavior.
    The system is defined by the following equations:

    ```math
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    ```
    """

    def __init__(
        self,
        length: int = 1000,
        target_delay: int = 1,
        input_dimensions: Optional[List[Literal["x", "y", "z"]]] = None,
        starting_point: Optional[Tuple[float, float, float]] = None,
        dt: float = 0.01,
        sigma: float = 10,
        beta: float = 8 / 3,
        rho: float = 28,
        return_full_sequence: bool = False,
    ):
        """Initialize Lorenz system dataset.

        Args:
            length (int, optional): Length of the time series. Defaults to 1000.
            target_delay (int, optional): Delay between input and target. Defaults to 1.
            input_dimensions (Optional[List[Literal["x", "y", "z"]]], optional): Dimensions to use as input. Defaults to None.
            starting_point (Optional[Tuple[float, float, float]], optional): Starting point of the system. Defaults to None.
            dt (float, optional): Time step. Defaults to 0.01.
            sigma (float, optional): Sigma parameter of the system. Defaults to 10.
            beta (float, optional): Beta parameter of the system. Defaults to 8/3.
            rho (float, optional): Rho parameter of the system. Defaults to 28.
            return_full_sequence (bool, optional): Return the full sequence or not. Defaults to False.
        """
        if input_dimensions is not None:
            if not all(dim in ["x", "y", "z"] for dim in input_dimensions):
                raise ValueError("input_dimensions must be a subset of ['x', 'y', 'z']")
        self.length = length
        self.target_delay = target_delay
        self.starting_point = starting_point or (0.1, 0.1, 0.1)
        self.input_dimensions = (
            sorted(input_dimensions)
            if input_dimensions is not None
            else ["x", "y", "z"]
        )
        self.dt = dt
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.return_full_sequence = return_full_sequence

        self.data, self.end_point = self._generate_data()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.return_full_sequence:
            return 1
        return self.length - self.target_delay

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the item at the given index.

        If return_full_sequence is True, the index is ignored.
        """
        data, target = [], []
        for dim in ["x", "y", "z"]:
            if self.return_full_sequence:
                if dim in self.input_dimensions:
                    data.append(self.data[dim][: -self.target_delay])
                target.append(self.data[dim][self.target_delay :])
            else:
                if dim in self.input_dimensions:
                    data.append(self.data[dim][idx])
                target.append(self.data[dim][idx + self.target_delay])

        return torch.stack(data, dim=-1), torch.stack(target, dim=-1)

    def _generate_data(self) -> Tuple[dict, Tuple[float, float, float]]:
        """Generate the data for the Lorenz system.

        Returns:
            Tuple[dict, Tuple[float, float, float]]: Generated data and the end point of the system.
        """
        x, y, z = self.starting_point
        data: Dict[str, List[float]] = {
            "x": [],
            "y": [],
            "z": [],
        }

        for _ in range(self.length):
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z

            x += dx * self.dt
            y += dy * self.dt
            z += dz * self.dt

            data["x"].append(x)
            data["y"].append(y)
            data["z"].append(z)

        max_value = max(
            max(np.abs(data["x"])), max(np.abs(data["y"])), max(np.abs(data["z"]))
        )
        return {
            name: torch.tensor(value) / max_value for name, value in data.items()
        }, (x, y, z)
