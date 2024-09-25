# from pathlib import Path

from ray.rllib.models import ModelCatalog
# from ray.tune import register_env

from models.dqn_model import DqnModel
from .utils import get_policy_weights_from_checkpoint, get_model_class, get_config_cls
# from uav_envs import UavEnvironment
# from uav_envs.wrappers.pretext_wrapper import PretextWrapper
# from uav_envs.wrappers.raster_wrapper import RasterWrapper

# base_dir = Path(__file__).parent.parent
# register_env(
#     "UavEnv",
#     lambda cfg: RasterWrapper(
#         PretextWrapper(UavEnvironment(**cfg["params"]), pretext_dir=f'{base_dir}/{cfg["pretext_dir"]}')
#     )
# )
ModelCatalog.register_custom_model("uav_encoder", DqnModel)
