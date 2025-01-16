import glob
from pathlib import Path

import ray
import yaml
from ray import air, tune

import common  # noqa
import src.uav_rl.uav_envs  # noqa
from src.uav_rl.rl.common import get_policy_weights_from_checkpoint, get_model_class, get_config_cls
from src.uav_rl.models.lpp2d_model import UavCustomModel  # noqa
from src.uav_rl.models.lpp2d_mlp import UavSimbaModel  # noqa

ckpt_dir = './ckpt'
train_one_step = False

base_dir = Path(__file__).parent.parent
run_cfg = yaml.load(open(f'../configs/uav-vec-dqn.yaml'), Loader=yaml.FullLoader)
# run_cfg = yaml.load(open(f'../configs/cartpole-sac.yaml'), Loader=yaml.FullLoader)
run_cfg['algo']['training'].update({
    '_enable_learner_api': False,
    'model': {
        # 'custom_model': get_model_class(run_cfg['algo']['name']),
        'custom_model': 'simba',
        'custom_model_config': run_cfg['model']
    }
})
if run_cfg['algo']['name'] == 'SAC':
    run_cfg['algo']['training'].update({
        'model': {},
        'policy_model_config': run_cfg['algo']['training']['model'],
        'q_model_config': run_cfg['algo']['training']['model'],
    })

'''
if args.evaluation_interval > 0:
    config
    .evaluation(
        evaluation_num_env_runners=args.evaluation_num_env_runners,
        evaluation_interval=args.evaluation_interval,
        evaluation_duration=args.evaluation_duration,
        evaluation_duration_unit=args.evaluation_duration_unit,
        evaluation_parallel_to_training=args.evaluation_parallel_to_training,
    )
    .reporting(
        metrics_num_episodes_for_smoothing=(args.num_gpus or 1),
        report_images_and_videos=False,
        report_dream_data=False,
        report_individual_batch_item_stats=False,
    )
'''

if __name__ == '__main__':
    config = (
        get_config_cls(run_cfg['algo']['name'])()
        .exploration(
            explore=True,
            exploration_config={
                "initial_epsilon": 0.9,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 200_000,
            },
        )
        .rl_module(_enable_rl_module_api=False)
        .framework(framework="torch")
        .training(**run_cfg['algo']['training'])
        .resources(
            num_gpus=1,
            # num_cpus_per_worker=2,
            # num_cpus_per_learner_worker=4,
        )
        .rollouts(
            num_rollout_workers=8,
            rollout_fragment_length=40,
            # sample_async=True,
        )
        .environment(**run_cfg['env'])
        .multi_agent(count_steps_by='agent_steps')
        # .evaluation(
        #     evaluation_num_workers=1,
        #     evaluation_interval=10,
        #     evaluation_duration="auto",
        #     evaluation_duration_unit="episodes",
        #     evaluation_parallel_to_training=True,
        # )
        # .reporting(
        #     metrics_num_episodes_for_smoothing=1,
        #     report_images_and_videos=True,
        #     report_dream_data=False,
        #     report_individual_batch_item_stats=False,
        # )
    )
    if run_cfg['train']['offline']:
        config = (
            config
            .offline_data(
                input_="dataset",
                input_config={
                    'format': 'json',
                    'paths': glob.glob(f'{base_dir}/data/astar-out/*.json'),
                },
            )
            .exploration(explore=False)
        )
    algo = config.build()
    if run_cfg['train']['load_weights']:
        algo.set_weights(get_policy_weights_from_checkpoint('./ckpt'))
    if train_one_step:
        results = algo.train()
    else:
        tuner = tune.Tuner(
            run_cfg['algo']['name'],
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                # stop=run_cfg['stop'],
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
