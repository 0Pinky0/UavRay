import numpy as np
import torch
from torch.utils.data import Dataset


class PretextDataset(Dataset):
    def __init__(self, dir: str) -> None:
        dtype = [('x', np.float32, (20, 2)), ('len', np.long)]
        self.dataset = np.memmap(dir, mode='r+', shape=(600,), dtype=dtype)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.dataset['x'][idx]), torch.tensor(self.dataset['len'][idx])
