import argparse
import glob
import json

import numpy as np

from uav_envs.wrappers.offline_data_wrapper import OfflineDataCollector
from uav_envs.uav_env_v7 import UavEnvironment


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


parser = argparse.ArgumentParser(description="Collect data from UAV environment")
parser.add_argument("--fixed_obstacle_number", type=int, default=4)
parser.add_argument("--dynamic_obstacle_number", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=100, help="Number of episodes to collect data"
)
parser.add_argument("--output_dir", type=str, default="data/")
parser.add_argument(
    "--episodes_per_file", type=int, default=50, help="Number of episodes per file"
)

args = parser.parse_args()

env = UavEnvironment(
    (800, 800),
    args.fixed_obstacle_number,
    args.dynamic_obstacle_number,
    intermediate_rewards=True,
)
env = OfflineDataCollector(
    env, episodes_per_file=args.episodes_per_file, save_dir=args.output_dir
)


for i in range(args.num_episodes):
    env.reset()
    terminate = truncate = False

    while not terminate and not truncate:
        _, _, terminate, truncate, _ = env.step(env.action_space.sample())

    print(f"Episode {i} finished")

env.close()


# loop through and convert to q_learning format
step_wise_data = []
for file in glob.glob(f"{args.output_dir}/*.json"):
    data = OfflineDataCollector.q_learning(file)

    for i in range(len(data["obs"])):
        step_data = []
        for key in ["obs", "action", "reward", "next_obs", "terminated"]:
            step_data.append(data[key][i])
        step_wise_data.append(step_data)

    # save the data
    with open(file.replace(".json", "_tuple.json"), "w") as f:
        json.dump(step_wise_data, f, cls=NumpyEncoder)
