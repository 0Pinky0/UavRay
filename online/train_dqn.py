# from ray.rllib.algorithms.dqn import DQN, DQNConfig
import json

from ray.tune import register_env

from online.dqn import D3QNConfig, D3QN
from uav_envs.uav_env_v7 import UAVEnvironment
from uav_envs.wrappers.action_wrapper import ActionSpaceWrapper
from uav_envs.wrappers.pretext_wrapper import PretextWrapper
from uav_envs.wrappers.raster_wrapper import RasterWrapper

pretext_dir = None
# if cfg.env.pretext_dir:
#     pretext_dir = f'{Path(__file__).parent.parent}/{cfg.env.pretext_dir}'
register_env(
    "UavEnv-v7",
    lambda cfg: ActionSpaceWrapper(
        RasterWrapper(PretextWrapper(UAVEnvironment(**cfg), pretext_dir=pretext_dir, device='cpu'))),
)

config = (
    D3QNConfig()
    .framework(framework="torch")
    .training(replay_buffer_config={
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    })
    .resources(num_gpus=1)
    .env_runners(num_env_runners=1)
    .environment("UavEnv-v7", env_config={
        "dimensions": (800, 800)
    })
)
algo = D3QN(config=config)
result = algo.train()
print(json.dumps(str(result), sort_keys=False, indent=4))
