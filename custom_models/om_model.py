from typing import Sequence, Dict, Union, List, Any

import gymnasium as gym
import torch
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.conv_encoder import ConvEncoder
from model.dueling_head import DuelingHead


class OpponentModelling(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 *,
                 cnn_channels: Sequence[int] = (32, 64, 64),
                 kernel_sizes: Sequence[int] = (3, 3, 3),
                 strides: Sequence[int] = (1, 1, 1),
                 hidden_dim: int = 256,
                 seq_len: int = 20,
                 latent_dim: int = 4,
                 embed_dim: int = 64):
        nn.Module.__init__(self)
        super(OpponentModelling, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        raster_shape = obs_space['raster'].shape
        obs_dim = obs_space['observation'].shape.n
        action_num = action_space.shape.n
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=obs_dim,
            vec_out=hidden_dim,
        )
        self.q_head = DuelingHead(hidden_dim, action_num)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.encoder = PretextEncoder(2, embed_dim, latent_dim, hidden_dim, seq_len)
        self.decoder = PretextDecoder(2, embed_dim, latent_dim, hidden_dim, seq_len)

    def forward(self, input_dict: dict[str, torch.Tensor], state, seq_lens):
        observation, raster = input_dict['observation'], input_dict['raster']
        embed = self.encoder(observation, raster)
        q_values = self.q_head(embed)
        return q_values

    def custom_loss(
            self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:
        true_act, each_seq_len = loss_inputs['opponent_pos'], loss_inputs['pos_len'].to('cpu')
        pred_act, z_mean, z_log_var = self.forward(true_act, each_seq_len)
        batch_size = true_act.size(0)
        if each_seq_len is not None:  # mask out the padded sequences from loss calculation
            mask = torch.zeros(batch_size, self.seq_len + 1)  # [1024, 21]
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


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class PretextEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            embed_dim: int = 64,
            latent_dim: int = 2,
            hidden_dim: int = 256,
            seq_len: int = 20,
    ):
        super(PretextEncoder, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        self.RNN = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        for name, param in self.RNN.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

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
        each_seq_len = each_seq_len.to('cpu')
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
    ):
        super(PretextDecoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.mlp_in = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        self.RNN = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        for name, param in self.RNN.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            # nn.Sigmoid(),
        )

    def forward(
            self,
            z_mean: torch.Tensor,
            z_var: torch.Tensor,
            resample: bool = False,
    ) -> torch.Tensor:
        batch_size = z_mean.size(0)
        token = -1 * torch.ones(batch_size, self.input_dim)
        hidden_state = None
        outputs = torch.zeros(self.seq_len, batch_size, self.input_dim)
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
