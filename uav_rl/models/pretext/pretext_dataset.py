import numpy as np
import torch
from torch.utils.data import Dataset


class PretextDataset(Dataset):
    def __init__(self, file_dir: str, len_dataset: int = 30000, seq_len: int = 20) -> None:
        dtype = [('x', np.float32, (seq_len, 2)), ('len', np.long)]
        self.dataset = np.memmap(file_dir, mode='r+', shape=(len_dataset,), dtype=dtype)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.dataset['x'][idx]), torch.tensor(self.dataset['len'][idx])
