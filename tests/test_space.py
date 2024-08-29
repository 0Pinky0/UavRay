# from ray.rllib.algorithms.dqn import DQN, DQNConfig
import json

from ray.tune import register_env

from online.dqn import D3QNConfig, D3QN
from uav_envs.uav_env_v7 import UavEnvironment
from uav_envs.wrappers.action_wrapper import ActionSpaceWrapper
from uav_envs.wrappers.pretext_wrapper import PretextWrapper
from uav_envs.wrappers.raster_wrapper import RasterWrapper

pretext_dir = None
env = ActionSpaceWrapper(
        RasterWrapper(PretextWrapper(UavEnvironment(), pretext_dir=pretext_dir, device='cpu')))
action_space = env.action_space
observation_space = env.observation_space
print(observation_space)

