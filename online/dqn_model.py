"""PyTorch model for DQN"""

from typing import Sequence
import gymnasium as gym
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict

torch, nn = try_import_torch()


class D3QNTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to provide dueling-Q functionality."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            q_hiddens: Sequence[int] = (256,),
            dueling: bool = False,
            dueling_activation: str = "relu",
            add_layer_norm: bool = False
    ):
        nn.Module.__init__(self)
        super(D3QNTorchModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.dueling = dueling
        ins = num_outputs

        advantage_module = nn.Sequential()
        value_module = nn.Sequential()

        # Dueling case: Build the shared (advantages and value) fc-network.
        for i, n in enumerate(q_hiddens):
            advantage_module.add_module(
                "dueling_A_{}".format(i),
                SlimFC(ins, n, activation_fn=dueling_activation),
            )
            value_module.add_module(
                "dueling_V_{}".format(i),
                SlimFC(ins, n, activation_fn=dueling_activation),
            )
            # Add LayerNorm after each Dense.
            if add_layer_norm:
                advantage_module.add_module(
                    "LayerNorm_A_{}".format(i), nn.LayerNorm(n)
                )
                value_module.add_module("LayerNorm_V_{}".format(i), nn.LayerNorm(n))
            ins = n

        # Actual Advantages layer (nodes=num-actions).
        advantage_module.add_module(
            "A", SlimFC(ins, action_space.n, activation_fn=None)
        )

        self.advantage_module = advantage_module

        # Value layer (nodes=1).
        if self.dueling:
            value_module.add_module(
                "V", SlimFC(ins, 1, activation_fn=None)
            )
            self.value_module = value_module
        self.inverse_dynamics = nn.Sequential(
            SlimFC(num_outputs * 2, 256, activation_fn=dueling_activation),
            SlimFC(256, action_space.n, activation_fn=None)
        )

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Args:
            model_out: Embedding from the model layers.

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        action_scores = self.advantage_module(model_out)
        logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
        return action_scores, logits, logits

    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""

        return self.value_module(model_out)
