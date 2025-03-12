import glob
from pathlib import Path

import ray
import yaml
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig, Algorithm

import common  # noqa
import src.uav_rl.uav_envs  # noqa
from src.uav_rl.rl.common import get_policy_weights_from_checkpoint, get_model_class, get_config_cls
from src.uav_rl.models.lpp2d_model import UavCustomModel  # noqa
from src.uav_rl.models.simba_encoder import UavSimbaModel  # noqa

ckpt_dir = './ckpt'
train_one_step = False
# is_test = True
is_test = False

base_dir = Path(__file__).parent.parent
run_cfg = yaml.load(open(f'../configs/uav-dqn.yaml'), Loader=yaml.FullLoader)


if __name__ == '__main__':
    algo = Algorithm.from_checkpoint('/home/wjl/ray_results/DQN_2025-01-19_17-30-04')
    if train_one_step:
        results = algo.train()
    else:
        tuner = tune.Tuner(
            # run_cfg['algo']['name'],
            algo,
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                # stop=run_cfg['stop'],
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=400,
                    checkpoint_at_end=True,
                ),
            ),
        )
        results = tuner.fit()
    print(results)
    ray.shutdown()
