from typing import SupportsFloat, Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
import torch


class RasterWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            state_size: tuple[int, int] = (384, 384),
            state_downsize: tuple[int, int] = (128, 128),
            use_sgcnn: bool = True,
    ):
        super().__init__(env)
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.use_sgcnn = use_sgcnn
        obs_shape = (4, *self.state_downsize,)
        if use_sgcnn:
            self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            obs_shape = (16, *(ds // 8 for ds in self.state_downsize))
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.env.observation_space.spaces['raster'] = gym.spaces.Box(
                low=0., high=1., shape=obs_shape, dtype=np.float32
            )
        else:
            self.env.observation_space = gym.spaces.Dict({
                'observation': self.env.observation_space,
                'raster': gym.spaces.Box(
                    low=0., high=1., shape=obs_shape, dtype=np.float32
                ),
            })
        self.map_distance = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        self.map_obstacle = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        self.map_trajectory = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)

    def step(
            self, action: WrapperActType
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        x_tm1, y_tm1 = self.env.uav.position
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_t, y_t = self.env.uav.position
        cv2.line(
            self.map_trajectory,
            (np.round(x_tm1).astype(np.int32), np.round(y_tm1).astype(np.int32)),
            (np.round(x_t).astype(np.int32), np.round(y_t).astype(np.int32)),
            (1.,),
        )
        if isinstance(obs, dict):
            obs['raster'] = self.get_raster()
        else:
            obs = {
                'observation': obs,
                'raster': self.get_raster(),
            }
        return obs, reward, terminated, truncated, info

    def get_raster(self) -> np.ndarray:
        map_opponent = np.zeros((self.env.dimensions[1], self.env.dimensions[0]), dtype=np.float32)
        for d_obstacle in self.env.d_obstacles:
            pts = np.array([[
                *((np.round(pt[0]).astype(np.int32),
                   np.round(pt[1]).astype(np.int32)) for pt in d_obstacle.get_hull())
            ]], dtype=np.int32)
            cv2.fillPoly(map_opponent, [pts], color=(1.,))
        self.last_map_opponent = 0.5 * self.last_map_opponent + 0.5 * map_opponent
        self.last_map_opponent = np.where(
            np.abs(self.last_map_opponent) < 0.5 ** (4 - 1),
            0.,
            self.last_map_opponent,
        )
        map_obstacle_ = self.map_obstacle
        for structure in self.env.obstacles.get_occur_structures():
            up_left = structure.pts[0]
            down_right = structure.pts[2]
            self.map_obstacle[up_left.y:down_right.y, up_left.x:down_right.x] = 1.

        obs = np.stack((self.map_distance,
                        map_obstacle_,
                        self.map_trajectory,
                        self.last_map_opponent), axis=-1)
        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs = cv2.copyMakeBorder(obs, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                 cv2.BORDER_CONSTANT, value=np.array((0., 1., 0., 0.)), )
        x_t, y_t = np.round(self.env.uav.x).astype(np.int32), np.round(self.env.uav.y).astype(np.int32)
        if x_t < 0:
            x_t = 0
        elif x_t >= self.env.dimensions[0]:
            x_t = self.env.dimensions[0] - 1
        if y_t < 0:
            y_t = 0
        elif y_t >= self.env.dimensions[1]:
            y_t = self.env.dimensions[1] - 1
        leftmost = round(y_t)
        rightmost = round(y_t + 2 * diag_r_int)
        upmost = round(x_t)
        bottommost = round(x_t + 2 * diag_r_int)
        obs_cropped = obs[leftmost:rightmost, upmost:bottommost, :]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), -self.env.uav.direction, 1.0)
        dst_size = 2 * diag_r_int
        delta_leftmost = int(diag_r_int - self.state_size[0] / 2)
        delta_rightmost = delta_leftmost + self.state_size[0]
        obs_rotated = cv2.warpAffine(obs_cropped, rotation_mat, (dst_size, dst_size))
        obs_rotated = obs_rotated[
                      delta_leftmost:delta_rightmost,
                      delta_leftmost:delta_rightmost,
                      :]
        obs_rotated_resize = cv2.resize(obs_rotated, self.state_downsize)
        obs = obs_rotated_resize.transpose(2, 0, 1)
        if self.use_sgcnn:
            obs = self.sgcnn(obs)
        return obs

    def sgcnn(self, obs):
        sgcnn_size = 16
        # obs_ = obs.transpose(1, 2, 0)
        obs_ = obs
        # obs_ = torch.from_numpy(obs).type(dtype=torch.float32)
        obs_list = []
        center_size = self.state_downsize[0] // 2
        with torch.no_grad():
            for _ in range(4):
                obs_list.append(obs_[
                                :,
                                (center_size - sgcnn_size // 2):(center_size + sgcnn_size // 2),
                                (center_size - sgcnn_size // 2):(center_size + sgcnn_size // 2),
                                ])
                obs_ = self.max_pool(torch.from_numpy(obs_)).numpy()
                center_size //= 2
        return np.concatenate(obs_list, axis=0, dtype=np.float32)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        map_apf = np.sqrt(
            (np.broadcast_to(np.arange(0, self.env.dimensions[0]),
                             shape=(self.env.dimensions[1], self.env.dimensions[0]))
             - self.env.uav.goal_position[0]) ** 2
            + (np.broadcast_to(np.arange(0, self.env.dimensions[1]),
                               shape=(self.env.dimensions[0], self.env.dimensions[1]))
               .swapaxes(0, 1) - self.env.uav.goal_position[1]) ** 2)
        self.map_distance = 0.998 ** map_apf
        self.map_obstacle = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        self.map_trajectory = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        self.last_map_opponent = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        for structure in self.env.obstacles.get_fixed_structures():
            up_left = structure.pts[0]
            down_right = structure.pts[2]
            self.map_obstacle[up_left.y:down_right.y, up_left.x:down_right.x] = 1.
        if isinstance(obs, dict):
            obs['raster'] = self.get_raster()
        else:
            obs = {
                'observation': obs,
                'raster': self.get_raster(),
            }
        return obs, info
