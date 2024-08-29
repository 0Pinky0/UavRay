# from ray.rllib.algorithms.dqn import DQN, DQNConfig
import json
from pathlib import Path

import yaml
from omegaconf import DictConfig
from ray.tune import register_env

# from online.dqn import D3QNConfig, D3QN
from uav_envs.uav_env_v7 import UavEnvironment
from uav_envs.wrappers.pretext_wrapper import PretextWrapper
from uav_envs.wrappers.raster_wrapper import RasterWrapper

base_dir = Path(__file__).parent.parent
env_cfg = yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader)

pretext_dir = None
if env_cfg.env.pretext_dir:
    pretext_dir = f'{Path(__file__).parent.parent}/{env_cfg.env.pretext_dir}'
register_env(
    "UavEnv",
    lambda cfg: RasterWrapper(
        # PretextWrapper(
        UavEnvironment(**cfg.env.params),
        #     pretext_dir=f'{Path(__file__).parent.parent}/{cfg.env.pretext_dir}' if cfg.env.pretext_dir else None,
        #     device='cpu'
        # )
    )
)

config = (
    D3QNConfig()
    .framework(framework="torch")
    .training(replay_buffer_config={
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.9,
        "prioritized_replay_beta": 0.6,
        "prioritized_replay_eps": 3e-6,
    })
    .resources(num_gpus=1)
    .env_runners(num_env_runners=1)
    .environment("UavEnv", env_config={
        "dimensions": (800, 800)
    })
)
algo = D3QN(config=config)
result = algo.train()
print(json.dumps(str(result), sort_keys=False, indent=4))
