import os

import gymnasium as gym
import numpy as np

try:
    import orjson as json
except:
    import json


def convert_np_data(data: dict):
    """Convert a dict containing numpy data to pure python data."""
    converted_data = {}
    for key, item in data.items():
        if isinstance(item, list):
            converted_data[key] = list(map(convert_np_data, item))
        elif isinstance(item, dict):
            converted_data[key] = convert_np_data(item)
        elif isinstance(item, np.ndarray):
            converted_data[key] = item.tolist()
        elif isinstance(item, np.generic):
            converted_data[key] = item.item()
        else:
            converted_data[key] = item
    return converted_data


class OfflineDataCollector(gym.Wrapper):
    def __init__(
        self,
        env,
        save_format: str = "json",
        episodes_per_file: int = 100,
        save_info: bool = False,
        save_dir: str = "data",
    ):
        super(OfflineDataCollector, self).__init__(env)
        self.save_format = save_format
        self.episodes_per_file = episodes_per_file
        self.save_info = save_info
        self.save_dir = save_dir
        self.episode_count = 0
        self.reset_data()

    def reset_data(self):
        self.data = {
            "obs": [],
            "action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "info": [],
        }

    def save_data(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        file_name = f"data_{self.episode_count // self.episodes_per_file}"
        file_path = os.path.join(self.save_dir, file_name)

        if self.save_format == "npz":
            np.savez(
                file_path,
                obs=np.array(self.data["obs"]),
                action=np.array(self.data["action"]),
                reward=np.array(self.data["reward"]),
                terminated=np.array(self.data["terminated"]),
                truncated=np.array(self.data["truncated"]),
                info=np.array(self.data["info"]) if self.save_info else [],
            )
        else:
            with open(f"{file_path}.json", "w") as f:
                if json.__name__ == "orjson":
                    f.write(
                        json.dumps(self.data, option=json.OPT_SERIALIZE_NUMPY).decode(
                            "utf-8"
                        )
                    )
                else:
                    f.write(json.dumps(convert_np_data(self.data)))

    def reset(self, **kwargs):
        if self.episode_count > 0 and self.episode_count % self.episodes_per_file == 0:
            self.save_data()
            self.reset_data()

        self.episode_count += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.data["obs"].append(obs)
        self.data["action"].append(action)
        self.data["reward"].append(reward)
        self.data["terminated"].append(terminated)
        self.data["truncated"].append(truncated)
        if self.save_info:
            self.data["info"].append(info)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.data["obs"]:  # Save any remaining data
            self.save_data()
        super().close()

    @staticmethod
    def q_learning(file_path: str, save_file: bool = False):
        """Convert an offline data file to a Q-learning data."""

        format = file_path.split(".")[-1]

        if format == "npz":
            data = np.load(file_path, allow_pickle=True)
            obs = data["obs"]
            action = data["action"]
            reward = data["reward"]
            terminated = data["terminated"]
            truncated = data["truncated"]
        elif format == "json":
            with open(file_path, "r") as f:
                data = json.loads(f.read())
            obs = np.array(data["obs"])
            action = np.array(data["action"])
            reward = np.array(data["reward"])
            terminated = np.array(data["terminated"])
            truncated = np.array(data["truncated"])
        else:
            raise ValueError("Unsupported file format.")

        q_learning_data = {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "terminated": [],
            "truncated": [],
        }

        for i in range(len(obs) - 1):
            q_learning_data["obs"].append(obs[i])
            q_learning_data["action"].append(action[i])
            q_learning_data["reward"].append(reward[i])
            q_learning_data["next_obs"].append(obs[i + 1])
            q_learning_data["terminated"].append(terminated[i])
            q_learning_data["truncated"].append(truncated[i])

        if save_file:
            if format == "npz":
                save_path = file_path.replace(".npz", "_q_learning.npz")
                np.savez(
                    save_path,
                    obs=np.array(q_learning_data["obs"]),
                    action=np.array(q_learning_data["action"]),
                    reward=np.array(q_learning_data["reward"]),
                    next_obs=np.array(q_learning_data["next_obs"]),
                    terminated=np.array(q_learning_data["terminated"]),
                    truncated=np.array(q_learning_data["truncated"]),
                )
            elif format == "json":
                save_path = file_path.replace(".json", "_q_learning.json")
                with open(save_path, "w") as f:
                    if json.__name__ == "orjson":
                        f.write(
                            json.dumps(
                                q_learning_data, option=json.OPT_SERIALIZE_NUMPY
                            ).decode("utf-8")
                        )
                    else:
                        f.write(json.dumps(convert_np_data(q_learning_data)))

        return q_learning_data
