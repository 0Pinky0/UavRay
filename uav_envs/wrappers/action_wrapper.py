import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperObsType


class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        if isinstance(self.env.action_space, gym.spaces.Sequence):
            self.nvec = self.env.action_space.feature_space
        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.nvec = self.env.action_space.nvec
        self.env.action_space = gym.spaces.Discrete(
            np.prod(self.nvec).astype(np.int32)
        )

    def action(self, action: int) -> WrapperObsType:
        speed_action, turn_action = action // self.nvec[-1], action % self.nvec[-1]
        converted_action = (speed_action, turn_action)
        if isinstance(self.env.action_space, gym.spaces.Sequence):
            converted_action = [converted_action]
        return converted_action
