from __future__ import annotations

import math
import random
from typing import Optional, NamedTuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import HumanRendering

from uav_envs.wrappers.raster_wrapper import RasterWrapper


class NumericalRange:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    @property
    def mode(self):
        return self.max - self.min


class UavAgentState(NamedTuple):
    x: float
    y: float
    direction: float

    @property
    def position(self):
        return self.x, self.y

    @property
    def position_discrete(self):
        return round(self.x), round(self.y)


class UavAgent:
    length = 43
    width = 20
    v_range = NumericalRange(0.0, 6.0)
    w_range = NumericalRange(-20.0, 20.0)
    nvec = (6, 9)

    def __init__(
            self,
            position: tuple[float, float] = (None, None),
            direction: float = None,
    ):
        self.x, self.y = position
        self.direction = direction
        self.state = UavAgentState(self.x, self.y, self.direction)

    @property
    def position(self):
        return self.x, self.y

    @property
    def convex_hull(self):
        return np.array([
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 0 + 30)),
             self.y + 1.0 * self.width * -math.sin(math.radians(self.direction + 0 + 30))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 90 + 45)),
             self.y + 1.0 * self.width * -math.sin(math.radians(self.direction + 90 + 45))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 180 + 45)),
             self.y + 1.0 * self.width * -math.sin(math.radians(self.direction + 180 + 45))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 270 + 60)),
             self.y + 1.0 * self.width * -math.sin(math.radians(self.direction + 270 + 60))),
        ])

    def control(self, acc: int | float, steer: int | float):
        assert type(acc) is type(steer)
        if isinstance(acc, int | np.int64):
            assert 0 <= acc < UavAgent.nvec[0]
            assert 0 <= steer < UavAgent.nvec[1]
            linear_velocity = (UavAgent.v_range.min
                               + (acc + 1) / (UavAgent.nvec[0]) * UavAgent.v_range.mode)
            angular_velocity = (UavAgent.w_range.min
                                + steer / (UavAgent.nvec[1] - 1) * UavAgent.w_range.mode)
        elif isinstance(acc, float | np.float32):
            assert UavAgent.v_range.min <= acc <= UavAgent.v_range.max
            assert UavAgent.w_range.min <= steer <= UavAgent.w_range.max
            linear_velocity = acc
            angular_velocity = steer
        else:
            raise TypeError('Invalid action type')
        self.direction = (self.direction + angular_velocity) % 360
        dx = linear_velocity * math.cos(math.radians(self.direction))
        dy = linear_velocity * -math.sin(math.radians(self.direction))
        self.x += dx
        self.y += dy
        self.state = UavAgentState(self.x, self.y, self.direction)
        return self.state

    def reset(self, position: tuple[float, float], direction: float):
        self.x, self.y = position
        self.direction = direction
        self.state = UavAgentState(self.x, self.y, self.direction)
        return self.state


class UavEnvironment(gym.Env):
    metadata = {
        "render_modes": [
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": 50,
    }

    def __init__(
            self,
            dimensions: tuple[int, int] = (350, 350),
            render_mode: str = None,
            state_pixels: bool = False,
            state_size: tuple[int, int] = (128, 128),
            state_downsize: tuple[int, int] = (128, 128),
            num_obstacles_range: tuple[int, int] = (0, 0),
            use_sgcnn: bool = False,
    ):
        super().__init__()
        # Environmental parameters
        self.dimensions = dimensions
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.num_obstacles_range = num_obstacles_range
        self.sgcnn_size = 16
        self.use_sgcnn = use_sgcnn

        # RL parameters
        if use_sgcnn:
            self.observation_space = gym.spaces.Box(
                low=0., high=1., shape=(16, self.sgcnn_size, self.sgcnn_size,), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0., high=1., shape=(4, *self.state_downsize,), dtype=np.float32
            )
        self.action_space = gym.spaces.MultiDiscrete(UavAgent.nvec)

        # Agents
        self.player = UavAgent()
        self.opponent = UavAgent()
        self.goal_position = None
        self.last_state = None
        self.map_distance = None
        self.map_obstacle = None
        self.map_trajectory = None
        self.t = None
        # Misc
        self.render_mode = render_mode
        self.state_pixels = state_pixels
        self.screen = None
        self.clock = None
        self.isopen = True

    @property
    def state_size_diag(self) -> tuple[int, int]:
        return (np.ceil(np.sqrt(2) * self.state_size[0]).astype(np.int32),
                np.ceil(np.sqrt(2) * self.state_size[1]).astype(np.int32))

    @property
    def dimensions_diag(self) -> tuple[int, int]:
        return (np.ceil(np.sqrt(2) * self.dimensions[0]).astype(np.int32),
                np.ceil(np.sqrt(2) * self.dimensions[1]).astype(np.int32))

    def step(self, action: tuple[int, int]):
        self.opponent.control(0., 0.)
        # self.opponent.control(0.5, 0.16)
        state = self.player.control(action[0], action[1])
        reward, done = self.reward_and_done(self.last_state, state)
        obs = self.observation(state)
        self.last_state = state
        return obs, reward, done, False, {}

    def sgcnn(self, obs: np.ndarray):
        obs_ = obs.transpose(1, 2, 0)
        obs_list = []
        center_size = self.state_downsize[0] // 2
        for _ in range(4):
            obs_s = obs_[
                    (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
                    (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
                    :, ]
            obs_ = cv2.resize(obs_, (center_size, center_size))
            center_size //= 2
            obs_list.append(obs_s)
        return np.concatenate(obs_list, axis=-1).transpose(2, 0, 1)

    def reward_and_done(
            self, last_state: UavAgentState, state: UavAgentState
    ) -> tuple[float, bool]:
        convex_hull = self.player.convex_hull
        crashed: bool = (not (
                (0 < convex_hull[:, 0])
                & (convex_hull[:, 0] < self.dimensions[0])
                & (0 < convex_hull[:, 1])
                & (convex_hull[:, 1] < self.dimensions[1])
        ).all() or np.logical_and(self.obs_ego_centric[:, :, 1], self.obs_ego_centric[:, :, 2]).any()
                         or np.logical_and(self.obs_ego_centric[:, :, 3], self.obs_ego_centric[:, :, 2]).any())
        if self.player.x < 0 or self.player.x > self.dimensions[0]:
            self.player.x = float(np.clip(self.player.x, 0, self.dimensions[0]))
        if self.player.y < 0 or self.player.y > self.dimensions[1]:
            self.player.y = float(np.clip(self.player.y, 0, self.dimensions[1]))
        x_t, y_t = last_state.position_discrete
        x_tp1, y_tp1 = state.position_discrete
        dist2goal = math.hypot(self.player.x - self.goal_position[0],
                               self.player.y - self.goal_position[1])
        if dist2goal <= UavAgent.length:
            reward_reach = 10.
            goal_reached = True
        else:
            reward_reach = 0.
            goal_reached = False
        if not crashed:
            reward_const = -1.
            reward_goal = (math.hypot(x_tp1 - self.goal_position[0], y_tp1 - self.goal_position[1])
                           - math.hypot(x_t - self.goal_position[0], y_t - self.goal_position[1]))
            if reward_goal < 0.:
                reward_goal *= 5.
            elif reward_goal > 0.:
                reward_goal *= 1.
            reward = (reward_const
                      + reward_goal
                      + reward_reach)
        else:
            reward = -100.
        self.t += 1
        time_out = self.t == 2000
        done = crashed or goal_reached or time_out
        cv2.line(self.map_trajectory, pt1=(x_t, y_t), pt2=(x_tp1, y_tp1), color=(1.,))
        return reward, done

    def observation(self, state: UavAgentState) -> np.ndarray:
        self.map_player = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        self.map_opponent = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        cv2.fillPoly(self.map_player, [self.player.convex_hull.round().astype(np.int32)], color=(1.,))
        cv2.fillPoly(self.map_opponent, [self.opponent.convex_hull.round().astype(np.int32)], color=(1.,))

        obs = np.stack((self.map_distance, self.map_obstacle, self.map_player, self.map_opponent), axis=-1)
        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs = cv2.copyMakeBorder(obs, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                 cv2.BORDER_CONSTANT, value=np.array((0., 1., 0., 0.)), )
        leftmost = round(self.player.y)
        rightmost = round(self.player.y + 2 * diag_r_int)
        upmost = round(self.player.x)
        bottommost = round(self.player.x + 2 * diag_r_int)
        obs_cropped = obs[leftmost:rightmost, upmost:bottommost, :]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), -self.player.direction, 1.0)
        dst_size = 2 * diag_r_int
        delta_leftmost = int(diag_r_int - self.state_size[0] / 2)
        delta_rightmost = delta_leftmost + self.state_size[0]
        obs_rotated = cv2.warpAffine(obs_cropped, rotation_mat, (dst_size, dst_size))
        obs_rotated = obs_rotated[
                      delta_leftmost:delta_rightmost,
                      delta_leftmost:delta_rightmost,
                      :]
        # _range = np.max(obs_rotated[:, :, 0]) - np.min(obs_rotated[:, :, 0])
        # obs_rotated[:, :, 0] = (obs_rotated[:, :, 0] - np.min(obs_rotated[:, :, 0])) / _range
        self.obs_ego_centric = obs_rotated
        obs_rotated_resize = cv2.resize(obs_rotated, self.state_downsize)
        obs = obs_rotated_resize.transpose(2, 0, 1)
        # self.obs_ego_centric = obs_rotated
        if self.use_sgcnn:
            obs = self.sgcnn(obs)
        return obs

    def render_map(self) -> np.ndarray:
        rendered_map = np.ones((self.dimensions[1], self.dimensions[0], 3), dtype=np.float32) * 255.
        rendered_map = np.where(
            np.expand_dims(self.map_obstacle, axis=-1),
            0.,
            rendered_map,
        )
        # rendered_map = np.where(
        #     np.expand_dims(self.map_distance, axis=-1),
        #     np.expand_dims(self.map_distance, axis=-1) * np.array((255., 0., 0.)),
        #     rendered_map,
        # )
        rendered_map = np.where(
            np.expand_dims(self.map_player, axis=-1) != 0,
            np.array((255., 0., 0.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.map_opponent, axis=-1) != 0,
            np.array((0., 0., 255.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.map_trajectory, axis=-1) != 0,
            np.array((0., 255., 255.)),
            rendered_map,
        )
        cv2.circle(rendered_map, (
            round(self.goal_position[0]),
            round(self.goal_position[1]),
        ), radius=UavAgent.length, color=(0., 255., 0.))
        return rendered_map

    def render_self(self) -> np.ndarray:
        # obs_rotated_resize = cv2.resize(self.obs_ego_centric, self.state_size_downsize)
        rendered_map = np.ones((self.state_size[1], self.state_size[0], 3), dtype=np.float32) * 255.
        rendered_map = np.where(
            np.expand_dims(self.obs_ego_centric[:, :, 1], axis=-1),
            0.,
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.obs_ego_centric[:, :, 0], axis=-1),
            np.expand_dims(self.obs_ego_centric[:, :, 0], axis=-1) * np.array((255., 0., 0.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.obs_ego_centric[:, :, 2], axis=-1) != 0,
            np.array((255., 0., 0.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.obs_ego_centric[:, :, 3], axis=-1) != 0,
            np.array((0., 0., 255.)),
            rendered_map,
        )
        return rendered_map

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Randomize goal position
        self.goal_position = (
            random.randint(UavAgent.length, self.dimensions[0] - UavAgent.length),
            random.randint(UavAgent.length, self.dimensions[1] - UavAgent.length),
        )
        # self.goal_position = (
        #     600,
        #     400,
        # )

        # Initialize player
        state = self.player.reset(
            position=(
                np.random.uniform(self.dimensions[0] * 0.1, self.dimensions[0] * 0.9),
                np.random.uniform(self.dimensions[1] * 0.1, self.dimensions[1] * 0.9),
            ),
            direction=np.random.uniform(0, 360),
        )
        # state = self.player.reset(
        #     position=(
        #         400,
        #         400,
        #     ),
        #     direction=0,
        # )
        self.last_state = state
        # Initialize opponents
        self.opponent.reset(
            position=(
                np.random.uniform(self.dimensions[0] * 0.3, self.dimensions[0] * 0.7),
                np.random.uniform(self.dimensions[1] * 0.3, self.dimensions[1] * 0.7),
            ),
            direction=np.random.uniform(0, 360),
        )
        while (math.hypot(self.player.x - self.opponent.x, self.player.y - self.opponent.y) < 1.2 * UavAgent.length
               or math.hypot(self.goal_position[0] - self.opponent.x,
                             self.goal_position[1] - self.opponent.y) < 1.2 * UavAgent.length):
            self.opponent.reset(
                position=(
                    np.random.uniform(self.dimensions[0] * 0.3, self.dimensions[0] * 0.7),
                    np.random.uniform(self.dimensions[1] * 0.3, self.dimensions[1] * 0.7),
                ),
                direction=np.random.uniform(0, 360),
            )
        # Initialize maps
        # map_apf = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.bool_)
        # # map_apf[self.goal_position[0], self.goal_position[1]] = True
        # map_apf[self.goal_position[1], self.goal_position[0]] = True
        map_apf = np.sqrt(
            (np.broadcast_to(np.arange(0, self.dimensions[0]), shape=(self.dimensions[1], self.dimensions[0]))
             - self.goal_position[0]) ** 2
            + (np.broadcast_to(np.arange(0, self.dimensions[1]), shape=(self.dimensions[0], self.dimensions[1]))
               .swapaxes(0, 1) - self.goal_position[1]) ** 2)
        self.map_distance = 0.998 ** map_apf
        self.map_trajectory = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.float32)
        # Randomize obstacles
        self.map_obstacle = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.bool_)
        num_obstacles = random.randint(*self.num_obstacles_range)
        current_obstacle_num = 0
        # self.obstacles = []
        while current_obstacle_num < num_obstacles:
            o_x = random.randint(0, 700)
            o_y = random.randint(0, 700)
            o_len = random.randint(10, 80)
            o_wid = random.randint(10, 80)
            dist2player = cv2.pointPolygonTest(np.expand_dims(np.array([
                (o_x, o_y),
                (o_x + o_len, o_y + o_wid),
                (o_x + o_len, o_y),
                (o_x, o_y + o_wid),
            ]), axis=-2), self.player.position, True)
            dist2goal = math.hypot(self.goal_position[0] - o_x, self.goal_position[1] - o_y)
            if dist2player < -1.2 * UavAgent.length and dist2goal > 85:
                current_obstacle_num += 1
                self.map_obstacle[o_y:(o_y + o_wid), o_x:(o_x + o_len)] = True
                # self.obstacles.append((o_x, o_y, o_len, o_wid))
        self.map_distance = np.where(
            self.map_obstacle,
            0.,
            self.map_distance
        )
        # Get observation
        obs = self.observation(state)
        self.t = 1
        return obs, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface(
                (self.state_size[0], self.state_size[1]) if self.state_pixels else (
                    self.dimensions[0], self.dimensions[1])
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state_pixels:
            img = self.render_self()
        else:
            img = self.render_map()
        surf = pygame.surfarray.make_surface(img)
        self.screen.blit(surf, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    # agent = UavAgent(position=(112, 112), direction=0)
    # print(agent.convex_hull)
    if_render = True
    episodes = 3
    env = UavEnvironment(
        render_mode='rgb_array' if if_render else None,
        state_pixels=True,
        # state_pixels=False,
    )
    env = HumanRendering(env)

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            # action = env.action_space.sample()
            # obs, reward, done, _, info = env.step(action)
            obs, reward, done, _, info = env.step((0, 4))
            print(reward)
            if if_render:
                env.render()

    env.close()
