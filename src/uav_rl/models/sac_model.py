from typing import Dict, Union, List

import gymnasium as gym
import torch
import torch.nn.functional as F
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from tensordict import TensorDict
from torch import nn

from src.uav_rl.models.templates.model_constructor import get_constructed_model
from src.uav_rl.models.modules.inverse_dynamics import InverseDynamic


class SacModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):
        nn.Module.__init__(self)
        super(SacModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        action_num = action_space.n
        custom_model_config = model_config['custom_model_config']
        self.use_inverse_dynamic = custom_model_config['use_inverse_dynamic']
        self.channels_last = custom_model_config['channels_last']
        hidden_dim: int = custom_model_config['encoder_config'][-1]['model_config']['out_features']

        self.encoder = get_constructed_model(custom_model_config['encoder_config'])
        self.fc_head = nn.Linear(hidden_dim, num_outputs)
        with torch.no_grad():
            dummy = torch.from_numpy(self.obs_space.sample()).unsqueeze(0)
            dummy = restore_original_dimensions(dummy, self.obs_space, 'torch')
            _ = self.forward(dummy)
        self.inverse_dynamics_model = InverseDynamic(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_num=action_num,
        )

    def forward(self, input_dict: dict[str, dict[str, torch.Tensor]], state=None, seq_lens=None):
        if isinstance(input_dict, dict) and 'obs' in input_dict:
            input_dict = input_dict['obs']
        if self.channels_last:
            input_dict = input_dict.permute(0, 3, 1, 2)
        input_dict = input_dict.to(torch.float32)
        if isinstance(input_dict, dict):
            input = TensorDict(input_dict)
        else:
            input = TensorDict({'observation': input_dict})
        embed = self.encoder(input)['logits']
        logits = self.fc_head(embed)
        return logits, state

    def custom_loss(
            self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:
        if self.use_inverse_dynamic:
            action_predict = self.inverse_dynamics(loss_inputs)
            action_t = loss_inputs['actions']
            if isinstance(action_t, list):
                action_t = torch.tensor(action_t, dtype=torch.int64).to(loss_inputs['obs'].device())
            action_t = action_t.to(torch.int64)
            actions_one_hot = F.one_hot(action_t, num_classes=self.action_num).to(torch.float32)
            self_supervised_loss = F.cross_entropy(action_predict, actions_one_hot)
            policy_loss[0] += 0.25 * self_supervised_loss
        return policy_loss

    def inverse_dynamics(self, loss_inputs: Dict[str, TensorType]):
        obs_t = loss_inputs['obs']
        if isinstance(obs_t, dict):
            obs_t = restore_original_dimensions(obs_t, self.obs_space, 'torch')
        obs_tp1 = loss_inputs['new_obs']
        if isinstance(obs_t, dict):
            obs_t = restore_original_dimensions(obs_tp1, self.obs_space, 'torch')
        embed_t, _ = self.forward(obs_t)
        embed_tp1, _ = self.forward(obs_tp1)
        predicted_action = self.inverse_dynamics_model(embed_t, embed_tp1)
        return predicted_action
