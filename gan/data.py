from pathlib import Path
from typing import Any, Callable, Type

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms

_Transform = Callable[[Any], Any]


class _MNISTLikeDataModule(LightningDataModule):
    def __init__(
        self,
        variant: Type[MNIST],
        data_dir: Path,
        batch_size: int,
        num_workers: int = 1,
        transform: _Transform | None = None,
    ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        if transform is None:
            transform = transforms.ToTensor()
        self._transform = transform

        self._dataset_cls = variant
        self._mnist_train = self._mnist_val = self._mnist_test = None

    def prepare_data(self) -> None:
        self._dataset_cls(str(self._data_dir), train=True, download=True)
        self._dataset_cls(str(self._data_dir), train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            mnist_full = self._dataset_cls(
                str(self._data_dir), train=True, transform=self._transform
            )
            self._mnist_train, self._mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test":
            self._mnist_test = self._dataset_cls(
                str(self._data_dir), train=False, transform=self._transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._mnist_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._mnist_val, batch_size=self._batch_size, num_workers=self._num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._mnist_test, batch_size=self._batch_size, num_workers=self._num_workers
        )


class MNISTDataModule(_MNISTLikeDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 1,
        transform: _Transform | None = None,
    ):
        super().__init__(MNIST, data_dir, batch_size, num_workers, transform)


class FashionMNISTDataModule(_MNISTLikeDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 1,
        transform: _Transform | None = None,
    ):
        super().__init__(FashionMNIST, data_dir, batch_size, num_workers, transform)
