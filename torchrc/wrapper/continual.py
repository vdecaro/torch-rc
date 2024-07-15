import torch, copy

from torch.utils.data import DataLoader, ConcatDataset
from torch_esn.data.continual import continual_benchmark
from torch_esn.data.util.seq_loader import seq_collate_fn
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from .base import ESNWrapper

from typing import List, Literal, Optional, Tuple, Union
from torch_esn.model.reservoir import Reservoir


class ContinualESNWrapper(ESNWrapper):
    def __init__(
        self,
        dataset: str,
        users: List[str],
        batch_size: int,
        strategy: Literal["naive", "joint", "replay"],
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.strategy = strategy

        self.benchmark = continual_benchmark(dataset, users)
        self.stream = list(self.benchmark.full_stream)
        if self.strategy == "replay":
            len_dataset = sum([len(ctx.dataset) for ctx in self.stream])
            self.buffer_size = int(len_dataset * 0.05)
            self.buffers = None

    def ip_step(
        self,
        context: int,
        reservoir: Reservoir,
        mu: float,
        sigma: float,
        eta: float,
        epochs: int = 1,
        device: Optional[str] = None,
    ) -> Reservoir:
        return super().ip_step(
            self.get_loader(context, mode=self.strategy),
            reservoir,
            mu=mu,
            sigma=sigma,
            eta=eta,
            epochs=epochs,
            device=device,
        )

    def ridge_step(
        self,
        context: int,
        reservoir: Reservoir,
        l2: Optional[List[float]] = None,
        perc_rec: Optional[float] = 1.0,
        alpha: Optional[float] = 1.0,
        prev_A: Optional[torch.Tensor] = None,
        prev_B: Optional[torch.Tensor] = None,
        with_readout: bool = True,
        device: Optional[str] = None,
    ):
        strategy = "joint_replay" if self.strategy == "replay" else self.strategy
        return super().ridge_step(
            self.get_loader(context, mode=strategy),
            reservoir,
            l2=l2,
            perc_rec=perc_rec,
            alpha=alpha,
            prev_A=prev_A,
            prev_B=prev_B,
            with_readout=with_readout,
            device=device,
        )

    def test_likelihood(
        self,
        context: int,
        reservoir: Reservoir,
        mu: float,
        sigma: float,
        device: Optional[str] = None,
    ) -> Tuple[float, int]:
        return super().test_likelihood(
            loader=self.get_loader(context),
            reservoir=reservoir,
            mu=mu,
            sigma=sigma,
            device=device,
        )

    def test_accuracy(
        self,
        context: int,
        readout: Union[torch.Tensor, List[torch.Tensor]],
        reservoir: Reservoir,
        device: Optional[str] = None,
    ) -> Tuple[float, int]:
        return super().test_accuracy(
            loader=self.get_loader(context),
            readout=readout,
            reservoir=reservoir,
            device=device,
        )

    def get_loader(self, context: int, mode: Optional[str] = None):
        if mode is None or mode == "naive":
            if context >= 0:
                dataset = self.stream[context].dataset
            else:
                dataset = ConcatDataset(
                    [ctx.dataset for ctx in self.stream[: (-context + 1)]]
                )

        elif mode == "replay":
            if self.buffers is None:
                self.buffers = [ReservoirSamplingBuffer(self.buffer_size)]
                for i, ctx in enumerate(self.stream[:-1]):
                    curr_buffer = copy.deepcopy(self.buffers[i])
                    curr_buffer.update_from_dataset(ctx.dataset)
                    self.buffers.append(curr_buffer)
            if context > 0:
                return ReplayDataLoader(
                    data=self.stream[context].dataset,
                    memory=self.buffers[context].buffer,
                    batch_size=self.batch_size,
                )
            else:
                dataset = self.stream[context].dataset

        elif mode == "joint":
            dataset = ConcatDataset([ctx.dataset for ctx in self.stream[: context + 1]])

        elif mode == "joint_replay":
            dataset = ConcatDataset(
                [self.stream[context].dataset, self.buffers[context].buffer]
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=seq_collate_fn(scenario="continual"),
        )

    def get_dataset_size(self, context: int, with_strategy: bool = False) -> int:
        if context >= 0:
            if with_strategy and self.strategy != "naive":
                if self.strategy == "replay":
                    return len(self.stream[context].dataset) + len(
                        self.buffers[context].buffer
                    )
                elif self.strategy == "joint":
                    return sum([len(ctx.dataset) for ctx in self.stream[: context + 1]])
            else:
                return len(self.stream[context].dataset)
        else:
            return sum([len(ctx.dataset) for ctx in self.stream[: (-context + 1)]])
