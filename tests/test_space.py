# from ray.rllib.algorithms.dqn import DQN, DQNConfig
from pathlib import Path

import yaml

from uav_envs import UavEnvironment
from wrappers.pretext_wrapper import PretextWrapper
from wrappers import RasterWrapper

base_dir = Path(__file__).parent.parent
env_cfg = yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader)['env']
pretext_dir = None
env = RasterWrapper(
    PretextWrapper(UavEnvironment(**env_cfg["params"]), pretext_dir=f'{base_dir}/{env_cfg["pretext_dir"]}')
)
action_space = env.action_space
observation_space = env.observation_space
print(observation_space)
