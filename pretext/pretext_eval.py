import torch
from torch.utils.data import DataLoader

from pretext.cvae import PretextVAE
from pretext.pretext_dataset import PretextDataset

epoch = 200
batch_size = 1024
seq_len = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PretextDataset(file_dir='data/pretext_dataset.npy')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = torch.load("data/pretext_vae.pt", map_location=device)
x, length = next(iter(dataloader))
x = x.to(device)
with torch.no_grad():
    y_hat, _, _ = model(x, each_seq_len=length)
print(x[0])
print(y_hat[0])
