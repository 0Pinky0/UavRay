from typing import Sequence

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

from model.conv_encoder import ConvEncoder
from model.dueling_head import DuelingHead
from model.inverse_dynamics import InverseDynamic

import gymnasium as gym


class ComplexInputQNet(TorchModelV2, nn.Module):
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
                 hidden_dim=256):
        nn.Module.__init__(self)
        super(ComplexInputQNet, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        raster_shape = obs_space['raster'].shape
        obs_dim = obs_space['observation'].shape.n
        action_num =  action_space.shape.n
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=obs_dim,
            vec_out=hidden_dim,
        )
        self.q_head = DuelingHead(hidden_dim, action_num)
        self.inverse_dynamics_model = InverseDynamic(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_num=action_num,
        )

    def forward(self, input_dict: dict[str, torch.Tensor], state, seq_lens):
        observation, raster = input_dict['observation'], input_dict['raster']
        embed = self.encoder(observation, raster)
        q_values = self.q_head(embed)
        return q_values

    def inverse_dynamics(self, s_t, r_t, s_tp1, r_tp1):
        embed_t = self.encoder(s_t, r_t)
        embed_tp1 = self.encoder(s_tp1, r_tp1)
        predicted_action = self.inverse_dynamics_model(embed_t, embed_tp1)
        return predicted_action

    def get_q_value_distributions(self, model_out):
        action_scores = self.advantage_module(model_out)
        logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
        return action_scores, logits, logits

    def get_state_value(self, model_out):
        return self.value_module(model_out)
