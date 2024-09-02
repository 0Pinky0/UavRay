from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models import ModelCatalog

from model.cnn_qnet_model import UavEncoder
from envs.uav_env_v7 import UavEnvironment
from envs.wrappers.raster_wrapper import RasterWrapper

ModelCatalog.register_custom_model("cnn_qnet", UavEncoder)
env = RasterWrapper(UavEnvironment())
action_space = env.action_space
obs_space = env.observation_space
model = ModelCatalog.get_model_v2(
    obs_space=obs_space,
    action_space=action_space,
    num_outputs=action_space.n,
    model_config={
            "custom_model": "cnn_qnet",
            "custom_model_config": {

            },
        },
    framework="torch",
    model_interface=DQNTorchModel,
    name="q_func",
    # q_hiddens=config["hiddens"],
    # dueling=config["dueling"],
    # num_atoms=config["num_atoms"],
    # use_noisy=config["noisy"],
    # v_min=config["v_min"],
    # v_max=config["v_max"],
    # sigma0=config["sigma0"],
    # add_layer_norm=add_layer_norm,
)