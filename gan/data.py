from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int, num_workers: int = 1):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._to_tensor = transforms.ToTensor()

        self._mnist_train = self._mnist_val = self._mnist_test = None

    def prepare_data(self) -> None:
        MNIST(str(self._data_dir), train=True, download=True)
        MNIST(str(self._data_dir), train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            mnist_full = MNIST(
                str(self._data_dir), train=True, transform=self._to_tensor
            )
            self._mnist_train, self._mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test":
            self._mnist_test = MNIST(
                str(self._data_dir), train=False, transform=self._to_tensor
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
