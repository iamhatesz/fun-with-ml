from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class FileDataset(TensorDataset):
    def __init__(self, file_path: Path, block_size: int):
        arr = np.fromfile(str(file_path), dtype=np.uint16).astype(np.int64)
        arr_size = (arr.size // block_size) * block_size + 1
        tokens = torch.from_numpy(arr[:arr_size])
        super().__init__(tokens[:-1].view(-1, block_size), tokens[1:].view(-1, block_size))


class MemoryMapDataset(Dataset):
    def __init__(self, file_path: Path, block_size: int):
        self._arr = np.memmap(str(file_path), dtype=np.uint16, mode="r")
        self._block_size = block_size

    def __len__(self) -> int:
        return len(self._arr) - self._block_size

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._arr[item : item + self._block_size].astype(np.int64),
            self._arr[item + 1 : item + self._block_size + 1].astype(np.int64),
        )


if __name__ == "__main__":
    ds = MemoryMapDataset(
        Path(__file__).parent / "data" / "wikitext" / "train.bin", block_size=1024
    )
    loader = DataLoader(ds, batch_size=8)
    print(next(iter(loader)))
