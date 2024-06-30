import torch
from torch.utils.data import DataLoader

from ray_uav.pretext.cvae import PretextVAE
from ray_uav.pretext.pretext_dataset import PretextDataset

epoch = 300
batch_size = 512
seq_len = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PretextDataset(dir='pretext_dataset.npy')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = PretextVAE(
    input_dim=2,
    seq_len=seq_len,
    num_layers=4,
    device=device,
)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=1e-9, total_iters=epoch)

for i in range(epoch):
    x, length = next(iter(dataloader))
    x = x.to(device)
    with torch.no_grad():
        y_hat, z = model(x, each_seq_len=length)
    optimizer.zero_grad()
    loss = model.vae_loss(x, each_seq_len=length)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f'Epoch {i} | Loss: {loss.item()}, lr: {scheduler.get_lr()}')
print(x[0])
print(y_hat[0])
