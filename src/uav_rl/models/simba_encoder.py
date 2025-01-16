import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, use_silu: bool = False):
        super().__init__()
        # fc1
        fc_layer_1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        nn.init.kaiming_normal_(fc_layer_1.weight)
        # fc2
        fc_layer_2 = nn.Linear(4 * hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(fc_layer_2.weight)
        # ffn
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            fc_layer_1,
            nn.SiLU() if use_silu else nn.ReLU(),
            fc_layer_2,
        )

    def forward(self, x):
        return x + self.ffn(x)


class UavSimbaModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Network parameters
        vec_dim = 8 + 42
        hidden_dim = num_outputs
        use_silu = False

        # Simba network
        fc_layer = nn.Linear(vec_dim, hidden_dim)
        nn.init.orthogonal_(fc_layer.weight, 1)
        self.encoder = nn.Sequential(
            fc_layer,
            ResidualBlock(hidden_dim, use_silu),
            ResidualBlock(hidden_dim, use_silu),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']['observation']
        embedding = self.encoder(obs)
        return embedding, state


ModelCatalog.register_custom_model('simba', UavSimbaModel)
