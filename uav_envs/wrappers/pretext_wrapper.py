from collections import deque

import gymnasium as gym
import numpy as np
import torch

from pretext.cvae import PretextVAE


class PretextWrapper(gym.ObservationWrapper):
    def __init__(self, env, pretext_dir: str = None, device: torch.device | str = 'cpu'):
        super().__init__(env)
        self.device = device
        if pretext_dir:
            self.cvae = torch.load(pretext_dir, map_location=device)
        else:
            self.cvae = PretextVAE(device=device)
        self.cvae.to(device).eval()
        self.opponent_num = self.env.obstacles.dynamic_number
        self.history = deque(maxlen=self.cvae.seq_len)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.env.observation_space.spaces['observation'] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.env.observation_space.spaces['observation'].shape[0]
                                                 + self.cvae.latent_dim,),
                dtype=np.float32
            )
        else:
            self.env.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0] + self.cvae.latent_dim,),
                dtype=np.float32
            )

    def observation(self, observation: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        if self.opponent_num > 0:
            self.history.append(
                [*(self.env.d_obstacles[i].position for i in range(self.opponent_num))]
            )
            with torch.no_grad():
                input = torch.tensor(self.history, dtype=torch.float32).to(self.device)
                if input.ndim == 3:
                    input = input.transpose(0, 1)
                each_seq_len = torch.ones([self.opponent_num], dtype=torch.long) * len(self.history)
                opponent_modeling = self.cvae.encoder.predict(
                    input, each_seq_len
                )
                opponent_modeling = opponent_modeling.mean(dim=0)
                opponent_modeling = opponent_modeling.cpu().numpy()
        else:
            opponent_modeling = np.zeros([self.cvae.latent_dim])
        if isinstance(observation, dict):
            observation['observation'] = np.concatenate([observation['observation'], opponent_modeling], axis=-1)
        else:
            observation = np.concatenate([observation, opponent_modeling], axis=-1)
        return observation
