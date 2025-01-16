import gymnasium as gym
from pathlib import Path

from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from .utils import get_policy_weights_from_checkpoint, get_model_class, get_config_cls
from src.uav_rl.uav_envs import UavEnvironment
# import uav_envs  # noqa
# from uav_envs.wrappers.pretext_wrapper import PretextWrapper
from src.uav_rl.uav_envs.wrappers.raster_wrapper import RasterWrapper

base_dir = Path(__file__).parent.parent
register_env(
    "UavEnv",
    lambda cfg: RasterWrapper(
        UavEnvironment(**cfg)
    )
)
register_env(
    "UavEnvVec",
    lambda cfg: UavEnvironment(**cfg)
)
# ModelCatalog.register_custom_model("uav_encoder", DqnModel)
