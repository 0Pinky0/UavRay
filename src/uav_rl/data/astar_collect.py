import os
from pathlib import Path

import numpy as np
import yaml
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter

from data.astar_actor import AstarActor
from uav_envs import UavEnvironment
from wrappers.pretext_wrapper import PretextWrapper
from wrappers import RasterWrapper

base_dir = Path(__file__).parent.parent
env_cfg = yaml.load(open(f'{base_dir}/configs/env_astar_config.yaml'), Loader=yaml.FullLoader)["env"]
max_eps = 100

if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    # writer = JsonWriter(
    #     os.path.join(ray._private.utils.get_user_temp_dir(), "astar-out")
    # )
    writer = JsonWriter(
        os.path.join(base_dir, "data", "astar-out"),
        # compress_columns=[],
    )

    env = RasterWrapper(
        PretextWrapper(
            UavEnvironment(**env_cfg["params"]),
            pretext_dir=f'{base_dir}/{env_cfg["pretext_dir"]}'
        )
    )
    actor = AstarActor(env, step_limit=2000)
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    for eps_id in range(max_eps):
        obs, info = env.reset()
        actor.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        terminated = truncated = False
        t = 0
        while not terminated and not truncated:
            # action = env.action_space.sample()
            action = actor.get_action()
            new_obs, rew, terminated, truncated, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                truncateds=truncated,
                infos={},
                new_obs=prep.transform(new_obs),
            )
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
            writer.write(batch_builder.build_and_reset())
        print(f"Episode ({eps_id} / {max_eps}) done due to {info['done']}")
