from typing import Union

from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.train import Checkpoint

from models.dqn_model import DqnModel
from models.ppo_model import PpoModel
from models.sac_model import SacModel

_model_class_mapping = {
    'DQN': DqnModel,
    'PPO': PpoModel,
    'SAC': SacModel,
}

_config_class_mapping = {
    'DQN': DQNConfig,
    'PPO': PPOConfig,
    'SAC': SACConfig,
}

def get_policy_weights_from_checkpoint(checkpoint: Union[str, Checkpoint],
                                       policy_name: str = 'default_policy'):
    loaded_weights = (
        Algorithm
        .from_checkpoint(checkpoint)
        .get_policy()
        .get_weights()
    )
    return {
        policy_name: loaded_weights
    }

def get_model_class(run_name: str) -> TorchModelV2:
    if run_name in _model_class_mapping:
        return _model_class_mapping[run_name]
    else:
        raise ValueError(f'Invalid run name: {run_name}')

def get_config_cls(run_name: str) -> AlgorithmConfig:
    if run_name in _config_class_mapping:
        return _config_class_mapping[run_name]
    else:
        raise ValueError(f'Invalid run name: {run_name}')
