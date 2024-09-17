import json
from pathlib import Path

import yaml
from ray.rllib import Policy
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from model.uav_encoder import UavEncoder
from envs.uav_env_v7 import UavEnvironment
from envs.wrappers.pretext_wrapper import PretextWrapper
from envs.wrappers.raster_wrapper import RasterWrapper

base_dir = Path(__file__).parent.parent.parent
env_cfg = yaml.load(open(f'{base_dir}/configs/env_online_config.yaml'), Loader=yaml.FullLoader)["env"]

register_env(
    "UavEnv",
    lambda cfg: RasterWrapper(
        PretextWrapper(UavEnvironment(**cfg["params"]), pretext_dir=f'{base_dir}/{cfg["pretext_dir"]}')
    )
)
ModelCatalog.register_custom_model("uav_encoder", UavEncoder)
hidden_dim = 256

config = (
    DQNConfig()
    .framework(framework="torch")
    .training(
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 60000,
            "prioritized_replay_alpha": 0.9,
            "prioritized_replay_beta": 0.6,
            "prioritized_replay_eps": 3e-6,
        },
        model={
            "custom_model": "uav_encoder",
            "custom_model_config": {
                'encoder_config': [
                    {
                        'model_name': 'conv_net',
                        'in_keys': ['raster'],
                        'out_keys': ['logits'],
                        'model_config': {
                            'num_channels': (16, 32, 64),
                            'kernel_sizes': 3,
                            'activation_class': 'relu',
                            'squash_last_layer': True,
                        },
                    },
                    {
                        'model_name': 'mlp',
                        'in_keys': ['logits', 'observation'],
                        'out_keys': ['logits'],
                        'model_config': {
                            'out_features': hidden_dim,
                            'activation_class': 'relu',
                            'activate_last_layer': True,
                        },
                    },
                ]
            },
        },
        double_q=True,
        dueling=True,
        n_step=3,
        hiddens=[hidden_dim, hidden_dim // 2],
    )
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=3)
    .environment("UavEnv", env_config=env_cfg)
)
# algo = config.build()
algo = DQN(config=config)
result = algo.train()
save_result = algo.save('./ckpt')
path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
print(json.dumps(str(result), sort_keys=False, indent=4))
