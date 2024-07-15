import os
from tqdm import tqdm
import numpy as np
from itertools import combinations
from math import comb

from torchvision.datasets import MNIST, SVHN, USPS, VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from scipy.io import loadmat

from typing import Any, Callable, Literal, Optional, Tuple

__all__ = ["Digit5"]


RANDOM_SEED = 42

# TODO: Augmentation of USPS to achieve balanced datasets (~60000 samples per dataset)


class Digit5(VisionDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        img_size: Tuple[int] = (3, 32, 32),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if split not in ["train", "test"]:
            raise ValueError("Parameter split must be in ['train', 'test'].")

        self.root = os.path.join(root, "digit5")
        os.makedirs(self.root, exist_ok=True)
        self.split = split
        self.img_size = img_size

        self.data, self.targets = [], []
        self.partition_idx = []
        self._load_data()

    def __len__(self) -> int:
        return sum([len(p) for p in self.partition_idx])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        part_idx = index
        for i, p in enumerate(self.partition_idx):
            if part_idx < len(p):
                img, label = self.data[i][p[part_idx]][0], self.targets[index]
                break
            part_idx -= len(p)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def _load_data(self) -> None:
        tt, min_max, resize, pad = (
            ToTensor(),
            lambda x: x / 255,
            Resize((self.img_size[1], self.img_size[2])),
            lambda x: x.repeat(3, 1, 1),
        )
        self.data = [
            LazyMNIST(
                self.root, split=self.split, transform=Compose([tt, resize, pad])
            ),
            LazySVHN(self.root, split=self.split),
            LazyUSPS(self.root, split=self.split, transform=Compose([tt, resize, pad])),
            LazySynthDigits(
                self.root, self.split, transform=Compose([tt, min_max, resize, pad])
            ),
            LazyMNISTM(
                self.root, self.split, transform=Compose([tt, min_max, resize, pad])
            ),
        ]
        self.targets = [d.targets for d in self.data]
        self.targets = [t for d_targets in self.targets for t in d_targets]
        self.partition_idx = [list(range(len(d))) for d in self.data]

    def apply_local_cluster_partition(
        self, client_index: int, n_clients: int, dataset_per_cluster: int
    ) -> None:

        # Indices is a list of lists of indices of each sub-dataset
        data_indices = [list(range(len(d))) for d in self.data]
        n_datasets = len(data_indices)
        if not (0 < dataset_per_cluster < n_datasets):
            raise ValueError(
                "The number of dataset per cluster must be in [1, 2, 3, 4, 5]."
            )
        if client_index >= n_clients:
            raise ValueError("Value of client_index must be < n_clients.")

        # n_partitions determines, for each subdataset, in how many clusters is involved
        n_partitions = (
            comb(n_datasets, dataset_per_cluster) * dataset_per_cluster // n_datasets
        )
        # Then, we split each sub-dataset to n_partitions chunks
        indices = []
        for i in data_indices:
            arr = np.array(i)
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(arr)
            indices.append(np.array_split(arr, n_partitions))

        # cluster_masks determines, for each cluster, what are the sub-dataset from
        # which they get a chunk
        cluster_masks = combinations(list(range(n_datasets)), r=dataset_per_cluster)

        # In this cycle, we distribute the chunks on the clusters
        cluster_indices = []
        for c_mask in cluster_masks:
            cluster_idx = [[] for _ in range(n_datasets)]
            for i in c_mask:
                cluster_idx[i] = indices[i].pop(0)
            cluster_indices.append(cluster_idx)

        # Ensure that all the chunks were distributed
        assert all([not idx for idx in indices])

        # Now we select the correct cluster for the client and we give it a partition
        # of the cluster's data
        n_clusters = len(cluster_indices)
        div, rem = n_clients // n_clusters, n_clients % n_clusters
        n_clients_per_cluster = []
        for _ in range(n_clusters):
            if rem > 0:
                n_clients_per_cluster.append(div + 1)
                rem -= 1
            else:
                n_clients_per_cluster.append(div)

        client_cluster = cluster_indices[client_index % n_clusters]
        n_clients_per_cluster = n_clients_per_cluster[client_index % n_clusters]
        client_indices = [[] for _ in range(n_datasets)]
        for i, sub_data_idx in enumerate(client_cluster):
            if sub_data_idx != []:
                client_indices[i] = (
                    np.array_split(sub_data_idx, n_clients_per_cluster)[
                        client_index % n_clients_per_cluster
                    ]
                ).tolist()

        self.indices = client_indices

    @property
    def indices(self):
        return self.partition_idx

    @indices.setter
    def indices(self, new_idx):
        if len(new_idx) != len(self.partition_idx):
            raise ValueError(
                "new_idx must contain a list of indices for each sub-dataset."
            )
        self.partition_idx = new_idx


class LazyVisionDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        self.split = split

        self.data, self.targets = [], []
        if not os.path.exists(os.path.join(self.fetch_root, self.split)):
            self._check_raw_data_availability()
            self._preprocess()
        self._load_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = (
            default_loader(os.path.join(self.fetch_root, self.split, self.data[index])),
            self.targets[index],
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def _load_data(self):
        with open(os.path.join(self.fetch_root, f"{self.split}_index.txt"), "r") as f:
            for line in f:
                path, label = line.split(" ")
                self.data.append(path)
                self.targets.append(int(label))

    def _check_raw_data_availability(self):
        raise NotImplementedError

    def _preprocess(self):
        raise NotImplementedError

    @property
    def fetch_root(self):
        raise NotImplementedError


class LazySynthDigits(LazyVisionDataset):
    def _check_raw_data_availability(self):
        if not os.path.exists(self.fetch_root):
            raise ValueError(f"Directory {self.fetch_root} not found.")

    def _preprocess(self):
        split_dir = os.path.join(self.fetch_root, self.split)
        os.makedirs(split_dir)
        mat = loadmat(os.path.join(self.fetch_root, f"synth_{self.split}_32x32.mat"))
        imgs, labels = mat["X"], mat["y"].astype(np.int64).squeeze()
        imgs = imgs.transpose((3, 0, 1, 2))
        for i in tqdm(
            range(len(imgs)), desc=f"Preprocessing synthdigits {self.split} split"
        ):
            f_name = f"{str(i).zfill(9)}.png"
            lab = labels[i].item()
            Image.fromarray(imgs[i]).save(os.path.join(split_dir, f_name), format="PNG")
            self.data.append(f_name)
            self.targets.append(lab)

        with open(os.path.join(self.fetch_root, f"{self.split}_index.txt"), "w+") as f:
            for f_name, lab in zip(self.data, self.targets):
                f.write(f"{f_name} {lab}\n")

    @LazyVisionDataset.fetch_root.getter
    def fetch_root(self):
        return os.path.join(self.root, "synthdigits")


class LazyMNISTM(LazyVisionDataset):
    def _check_raw_data_availability(self):
        if not os.path.exists(self.fetch_root):
            raise ValueError(f"Directory mnistm not found in path {self.root}.")

    def _preprocess(self):
        pass

    @LazyVisionDataset.fetch_root.getter
    def fetch_root(self):
        return os.path.join(self.root, "mnistm")


class LazySVHN(LazyVisionDataset):
    def _check_raw_data_availability(self):
        return True

    def _preprocess(self):
        split_dir = os.path.join(self.fetch_root, self.split)
        os.makedirs(split_dir)
        dataset = SVHN(self.fetch_root, split=self.split, download=True)
        for i in tqdm(
            range(len(dataset)), desc=f"Preprocessing SVHN {self.split} split"
        ):
            f_name = f"{str(i).zfill(9)}.png"
            img, lab = dataset[i]
            img.save(os.path.join(split_dir, f_name), format="PNG")
            self.data.append(f_name)
            self.targets.append(lab)

        with open(os.path.join(self.fetch_root, f"{self.split}_index.txt"), "w+") as f:
            for f_name, lab in zip(self.data, self.targets):
                f.write(f"{f_name} {lab}\n")

    @LazyVisionDataset.fetch_root.getter
    def fetch_root(self):
        return os.path.join(self.root, "svhn")


class LazyUSPS(LazyVisionDataset):
    def _check_raw_data_availability(self):
        return True

    def _preprocess(self):
        split_dir = os.path.join(self.fetch_root, self.split)
        os.makedirs(split_dir)
        dataset = USPS(self.fetch_root, train=(self.split == "train"), download=True)
        for i in tqdm(
            range(len(dataset)), desc=f"Preprocessing USPS {self.split} split"
        ):
            f_name = f"{str(i).zfill(9)}.png"
            img, lab = dataset[i]
            img.save(os.path.join(split_dir, f_name), format="PNG")
            self.data.append(f_name)
            self.targets.append(lab)

        with open(os.path.join(self.fetch_root, f"{self.split}_index.txt"), "w+") as f:
            for f_name, lab in zip(self.data, self.targets):
                f.write(f"{f_name} {lab}\n")

    @LazyVisionDataset.fetch_root.getter
    def fetch_root(self):
        return os.path.join(self.root, "usps")


class LazyMNIST(LazyVisionDataset):
    def _check_raw_data_availability(self):
        return True

    def _preprocess(self):
        split_dir = os.path.join(self.fetch_root, self.split)
        os.makedirs(split_dir)
        dataset = MNIST(self.root, train=(self.split == "train"), download=True)
        for i in tqdm(
            range(len(dataset)), desc=f"Preprocessing MNIST {self.split} split"
        ):
            f_name = f"{str(i).zfill(9)}.png"
            img, lab = dataset[i]
            img.save(os.path.join(split_dir, f_name), format="PNG")
            self.data.append(f_name)
            self.targets.append(lab)

        with open(os.path.join(self.fetch_root, f"{self.split}_index.txt"), "w+") as f:
            for f_name, lab in zip(self.data, self.targets):
                f.write(f"{f_name} {lab}\n")

    @LazyVisionDataset.fetch_root.getter
    def fetch_root(self):
        return os.path.join(self.root, "MNIST")
