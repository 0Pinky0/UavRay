import json
from pathlib import Path

import yaml
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
                'hidden_dim': hidden_dim,
                "raster_shape": (16, 16, 16),
                "vec_dim": 52,
                "cnn_channels": (32, 64, 128),
                "kernel_sizes": (3, 3, 3),
                "strides": (1, 1, 1),
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
print(json.dumps(str(result), sort_keys=False, indent=4))
