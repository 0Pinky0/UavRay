import glob
from pathlib import Path

import ray
import yaml
from ray import air, tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from uav_envs import UavEnvironment
from uav_envs.wrappers.pretext_wrapper import PretextWrapper
from uav_envs.wrappers.raster_wrapper import RasterWrapper
from models.uav_encoder import UavEncoder

ckpt_dir = './ckpt'
load_weights = False
is_offline = False
train_one_step = False

base_dir = Path(__file__).parent.parent
env_cfg = yaml.load(open(f'{base_dir}/configs/env/env_online_config.yaml'), Loader=yaml.FullLoader)["env"]

register_env(
    "UavEnv",
    lambda cfg: RasterWrapper(
        PretextWrapper(UavEnvironment(**cfg["params"]), pretext_dir=f'{base_dir}/{cfg["pretext_dir"]}')
    )
)
ModelCatalog.register_custom_model("uav_encoder", UavEncoder)
hidden_dim = 256


def get_policy_weights_from_checkpoint(checkpoint):
    loaded_weights = Algorithm.from_checkpoint(checkpoint).get_policy().get_weights()
    return {
        'default_policy': loaded_weights
    }

if __name__ == '__main__':
    config = (
        DQNConfig()
        .framework(framework="torch")
        .training(
            double_q=True,
            dueling=True,
            n_step=3,
            hiddens=[hidden_dim, hidden_dim // 2],
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
                    'use_inverse_dynamic': False,
                    'encoder_config': [
                        # {
                        #     'model_name': 'conv_net',
                        #     'in_keys': ['raster'],
                        #     'out_keys': ['logits'],
                        #     'model_config': {
                        #         'num_channels': (16, 32, 64),
                        #         'kernel_sizes': 3,
                        #         'activation_class': 'relu',
                        #         'squash_last_layer': True,
                        #     },
                        # },
                        {
                            'model_name': 'mlp',
                            'in_keys': ['observation'],
                            # 'in_keys': ['logits', 'observation'],
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
        )
        .resources(num_gpus=1)
        .rollouts(num_rollout_workers=3)
        # .environment("UavEnv", env_config=env_cfg)
        .environment("CartPole")
    )
    if is_offline:
        config = (
            config
            .offline_data(
                input_="dataset",
                input_config={
                    "format": "json",
                    "paths": glob.glob(f"{base_dir}/data/astar-out/*.json"),
                },
            )
            .exploration(explore=False)
        )

    algo = config.build()
    if load_weights:
        algo.set_weights(get_policy_weights_from_checkpoint('./ckpt'))
    if train_one_step:
        results = algo.train()
    else:
        stop = {
            "training_iteration": 10_000,
            "timesteps_total": 100_000,
            "episode_reward_mean": 1000.0,
        }
        # stop = {}
        tuner = tune.Tuner(
            "DQN",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop=stop,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=5,
                    checkpoint_at_end=True,
                ),
            ),
        )
        results = tuner.fit()
    print(results)
    ray.shutdown()
