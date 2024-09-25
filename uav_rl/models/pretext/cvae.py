from typing import Union, Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), 2 ** 0.5)


class PretextVAE(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            embed_dim: int = 64,
            latent_dim: int = 2,
            hidden_dim: int = 256,
            seq_len: int = 20,
            num_layers: int = 1,
            conditioned: bool = True,
            device: torch.device = 'cpu',
    ):
        super(PretextVAE, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.encoder = PretextEncoder(input_dim, embed_dim, latent_dim, hidden_dim, seq_len, num_layers, device)
        self.decoder = PretextDecoder(input_dim, embed_dim, latent_dim, hidden_dim, seq_len, num_layers, conditioned, device)

    def forward(
            self,
            x: torch.Tensor,
            each_seq_len: Union[Any, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_var = self.encoder(x, each_seq_len)
        y = self.decoder(z_mean, z_var)
        return y, z_mean, z_var

    def vae_loss(self, true_act: torch.Tensor, each_seq_len: Union[Any, torch.Tensor] = None) -> torch.Tensor:
        pred_act, z_mean, z_log_var = self.forward(true_act, each_seq_len)
        batch_size = true_act.size(0)
        if each_seq_len is not None:  # mask out the padded sequences from loss calculation
            mask = torch.zeros(batch_size, self.seq_len + 1).to(self.device)  # [1024, 21]
            mask[torch.arange(batch_size), each_seq_len] = 1.
            mask = torch.logical_not(mask.cumsum(dim=1))
            # remove the sentinel
            mask = mask[:, :-1]  # [1024, 20]
            # BCE = F.gaussian_nll_loss(act_mean, true_act, act_var)
            BCE = F.mse_loss(pred_act[mask], true_act[mask]) * 10
        else:
            BCE = F.mse_loss(pred_act, true_act) * 10
        KLD = -torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        beta = 5e-7
        return BCE + KLD * beta


class PretextEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            embed_dim: int = 64,
            latent_dim: int = 2,
            hidden_dim: int = 256,
            seq_len: int = 20,
            num_layers: int = 1,
            device: torch.device = 'cpu',
    ):
        super(PretextEncoder, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.embedding = nn.Sequential(
            init_(nn.Linear(input_dim, 32)), nn.ReLU(),
            init_(nn.Linear(32, embed_dim)), nn.ReLU(),
        ).to(self.device)
        self.RNN = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers).to(self.device)
        for name, param in self.RNN.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.output_linear = nn.Sequential(
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, latent_dim * 2)),
        ).to(self.device)

    def forward(
            self,
            x: torch.Tensor,
            each_seq_len: Union[Any, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.embedding(x)
        if each_seq_len is not None:
            if not isinstance(each_seq_len, torch.Tensor):
                if isinstance(each_seq_len, int):
                    each_seq_len = [each_seq_len]
                each_seq_len = torch.tensor(each_seq_len, dtype=torch.long)
        else:
            each_seq_len = torch.ones([x.size(0)], dtype=torch.long) * self.seq_len
        x = pack_padded_sequence(x, each_seq_len, batch_first=True, enforce_sorted=False)
        _, hidden_state = self.RNN(x)
        hidden_state = hidden_state[-1]
        mean, var = self.output_linear(hidden_state).chunk(2, -1)
        return mean, var

    def predict(
            self,
            x: torch.Tensor,
            each_seq_len: Union[Any, torch.Tensor] = None,
    ) -> torch.Tensor:
        mean, var = self.forward(x, each_seq_len)
        z = reparameterize(mean, var)
        return z


class PretextDecoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            embed_dim: int = 64,
            latent_dim: int = 2,
            hidden_dim: int = 256,
            seq_len: int = 20,
            num_layers: int = 1,
            conditioned: bool = True,
            device: torch.device = 'cpu',
    ):
        super(PretextDecoder, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.input_dim = input_dim
        self.conditioned = conditioned
        self.mlp_in = nn.Sequential(
            init_(nn.Linear(conditioned * input_dim + latent_dim, 32)), nn.ReLU(),
            init_(nn.Linear(32, embed_dim)), nn.ReLU(),
        ).to(self.device)
        self.RNN = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers).to(self.device)
        for name, param in self.RNN.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.output_linear = nn.Sequential(
            init_(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim // 2, input_dim)),
            # nn.Sigmoid(),
        ).to(self.device)

    def forward(
            self,
            z_mean: torch.Tensor,
            z_var: torch.Tensor,
            resample: bool = False,
    ) -> torch.Tensor:
        batch_size = z_mean.size(0)
        token = -1 * torch.ones(batch_size, self.input_dim).to(self.device)
        hidden_state = None
        outputs = torch.zeros(self.seq_len, batch_size, self.input_dim).to(self.device)
        if not resample:
            z = reparameterize(z_mean, z_var)
        for i in range(self.seq_len):
            if resample:
                z = reparameterize(z_mean, z_var)
            if self.conditioned:
                input = torch.cat((token, z), dim=-1)
            else:
                input = z
            input = self.mlp_in(input)
            token, hidden_state = self.RNN(input.unsqueeze(0), hidden_state)
            token = self.output_linear(token.squeeze(0))
            outputs[i] = token
        outputs = outputs.permute(1, 0, 2)
        return outputs


if __name__ == '__main__':
    vae = PretextVAE(latent_dim=4)
    batch = 5
    fake_data = torch.zeros([batch, 20, 2])
    output, z = vae(fake_data, [20] * batch)
    print(output.shape)
    embed = vae.encoder.predict(fake_data, [20] * batch)
    print(embed.shape)
