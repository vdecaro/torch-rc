from typing import List, Literal, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class LorenzSystem(Dataset):
    """Lorenz system dataset."""

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

    def __len__(self):
        if self.return_full_sequence:
            return 1
        return self.length - self.target_delay

    def __getitem__(self, idx: int):
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

    def _generate_data(self):
        x, y, z = self.starting_point
        data = {
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
