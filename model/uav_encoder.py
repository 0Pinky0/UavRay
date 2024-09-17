from typing import Dict, Union, List

import gymnasium as gym
import torch
import torch.nn.functional as F
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from tensordict import TensorDict
from torch import nn

from model.model_constructor import get_constructed_model
from model.modules.inverse_dynamics import InverseDynamic


class UavEncoder(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):
        nn.Module.__init__(self)
        super(UavEncoder, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        action_num = action_space.n
        self.action_num = action_num
        custom_model_config = model_config['custom_model_config']
        hidden_dim: int = custom_model_config.get('hidden_dim', 256)

        self.encoder = get_constructed_model(custom_model_config['encoder_config'])
        with torch.no_grad():
            dummy = torch.from_numpy(self.obs_space.sample()).unsqueeze(0)
            dummy = restore_original_dimensions(dummy, self.obs_space, 'torch')
            dummy = TensorDict(dummy)
            _ = self.encoder(dummy)
        self.inverse_dynamics_model = InverseDynamic(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_num=action_num,
        )

    def forward(self, input_dict: dict[str, dict[str, torch.Tensor]], state=None, seq_lens=None):
        if isinstance(input_dict['obs'], dict):
            input = TensorDict(input_dict['obs'])
        else:
            input = TensorDict({'observation': input_dict['obs']})
        embed = self.encoder(input)['logits']
        return embed, state

    def custom_loss(
            self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:
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
        obs_t = TensorDict(
            restore_original_dimensions(loss_inputs['obs'], self.obs_space, 'torch')
        )
        obs_tp1 = TensorDict(
            restore_original_dimensions(loss_inputs['new_obs'], self.obs_space, 'torch')
        )
        embed_t = self.encoder(obs_t)
        embed_tp1 = self.encoder(obs_tp1)
        predicted_action = self.inverse_dynamics_model(embed_t['logits'], embed_tp1['logits'])
        return predicted_action
