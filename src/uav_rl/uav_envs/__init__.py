import gymnasium.envs.registration
from .uav_env_v7 import UavEnvironment

gymnasium.envs.registration.register(
    id="UavEnv-v7",
    entry_point="uav_envs.uav_env_v7:UavEnvironment",
)
