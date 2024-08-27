from __future__ import annotations

import math
import os
import random

import cv2
import gymnasium as gym
import numpy as np
import pygame as pg
import pygame.freetype as pgft
from gymnasium.utils import seeding

root = os.path.dirname(os.path.abspath(__file__))
uav = pg.image.load(os.path.join(root, "res/uav2.jpg"))
uav = pg.transform.rotozoom(uav, 0, 0.22)
background = pg.image.load(os.path.join(root, "res/sky.png"))
background = pg.transform.rotozoom(background, 0, 2)
imagerect = uav.get_rect()

pgft.init()
font = pgft.SysFont("", 20)
excl_mark, _ = font.render("!!!", (255, 0, 0))

STATE_SIZE = 11
NUM_OBSTACLES = 4
MAX_OBSTACLE_LEN = 100
MAX_OBSTACLE_WID = 40


def angular_offset(uav: "UAV", target: "tuple[int, int]"):
    v_target = np.array(target)
    v_uav = uav.position

    vdesired = v_target - v_uav
    theta = math.atan2(
        math.sin(math.radians(uav.direction)), math.cos(math.radians(uav.direction))
    )
    thetatarget = math.atan2(vdesired[1], vdesired[0])
    thetadesired = math.degrees(thetatarget + theta)
    if math.fabs(thetadesired) > math.fabs(thetadesired + 360):  # 负
        thetadesired += 360
    if math.fabs(thetadesired) > math.fabs(thetadesired - 360):
        thetadesired -= 360
    thetadesired = -thetadesired
    return int(thetadesired)


class UAV:
    oxy = []
    max_speed = 1
    min_ship_spd = 0.1
    max_ang_vel = 0.3
    length = 86
    width = 40
    CRUISE = 10

    def __init__(
        self,
        pos: "tuple[int, int]",
        env: "UAVEnvironment",
        other_uav: "UAV" = None,
        initial_speed: float = 0,
        initial_direction: float = 0,
        initial_angular_velocity: float = 0,
        goal_position: "tuple[int, int]" = (200, 200),
    ):
        self.env = env

        # constants
        self.length = UAV.length

        # Internal state variables
        self.x, self.y = pos
        self.dx = 0
        self.dy = 0
        self.speed = initial_speed
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity

        # Episode dependent variables
        self.goal_position = goal_position
        self.other_ship = other_uav
        self.show_circles = False
        self.arrive = 0

        self.tracex = []
        self.tracey = []
        self.angular = 0
        self.angular_to_goal = 0
        self.time = 0
        self.t = 0
        self.obstacle = []

    @property
    def position(self):
        return (self.x, self.y)

    def update_state(self):
        other_ship_dist = math.hypot(
            self.x - self.other_ship.x, self.y - self.other_ship.y
        )
        dist_to_goal = math.hypot(
            self.x - self.goal_position[0], self.y - self.goal_position[1]
        )

        self.prev_dist = dist_to_goal
        goal_offset = angular_offset(self, self.goal_position)
        collision_offset = angular_offset(self, self.other_ship.position)

        other_goal_offset = angular_offset(
            self.other_ship, self.other_ship.goal_position
        )
        other_collision_offset = angular_offset(self.other_ship, self.position)
        g_nowx, g_nowy = self.goal_position[0] - self.x, self.goal_position[1] - self.y
        self.angular_to_goal = math.degrees(math.atan2(g_nowy, g_nowx))
        dist_to_obstacle = 0
        for i in range(0, NUM_OBSTACLES * 2, 2):
            dist_to_obstacle += math.hypot(self.x - UAV.oxy[i], self.y - UAV.oxy[i + 1])
        self.state = [
            self.speed / UAV.max_speed,
            self.angular_velocity / UAV.max_ang_vel,
            goal_offset / 180,
            dist_to_goal / self.env.dimensions[0],
            other_ship_dist / self.env.dimensions[1],
            other_ship_dist,
            collision_offset / 180,
            other_goal_offset / 180,
            other_collision_offset / 180,
            self.other_ship.speed,
            dist_to_obstacle,
        ]

    def update_obstacle(self):
        obs_x, obs_y = 0, 0
        obstacle = []
        random_len = [
            random.randint(math.ceil(self.length), math.ceil(self.length * 2.5))
            for _ in range(NUM_OBSTACLES)
        ]
        random_angle = [
            random.randint(int(self.direction - 135), int(self.direction - 45))
            for _ in range(NUM_OBSTACLES)
        ]
        for i, j in zip(random_len, random_angle):
            obs_x = round(self.x + i * math.cos(math.radians(self.direction + j)))
            obs_y = round(self.y + i * -math.sin(math.radians(self.direction + j)))
            obstacle.append((obs_x, obs_y))
        return obstacle

    def step(self, action: "tuple[int, int]"):
        done = False
        illegal_action = False

        if isinstance(action, (tuple, list, np.ndarray)):
            speed_action, turn_action = action
            # Define the conversion factors
            increments_per_grid = 1
            speed_increment = 0.1
            turn_increment = 1

            # Map the speed action to the speed increment
            if speed_action == 0:
                self.speed = 1 * increments_per_grid * speed_increment
            elif speed_action == 1:
                self.speed = 2 * increments_per_grid * speed_increment
            elif speed_action == 2:
                self.speed = 3 * increments_per_grid * speed_increment

            # Map the turn action to the angular increment
            if turn_action == 0:
                self.angular_velocity = -2 * increments_per_grid * turn_increment
            elif turn_action == 1:
                self.angular_velocity = -1 * increments_per_grid * turn_increment
            elif turn_action == 2:
                self.angular_velocity = 0
            elif turn_action == 3:
                self.angular_velocity = 1 * increments_per_grid * turn_increment
            elif turn_action == 4:
                self.angular_velocity = 2 * increments_per_grid * turn_increment
        elif isinstance(action, int) and action == UAV.CRUISE:
            self.speed = 0.5
            self.angular += 1
            self.angular_velocity = 0.16
            increments_per_grid = 1

        # Move the agent based on speed and direction
        for _ in range(increments_per_grid):
            self.dx = self.speed * math.cos(math.radians(self.direction))
            self.dy = self.speed * -math.sin(math.radians(self.direction))
            self.x += self.dx
            self.y += self.dy
            self.direction = (self.direction + self.angular_velocity) % 360
            self.tracex.append(self.x)
            self.tracey.append(self.y)

        dist_to_goal = math.hypot(
            self.x - self.goal_position[0], self.y - self.goal_position[1]
        )

        reward = (
            float((self.prev_dist - dist_to_goal))
            if self.env.intermediate_rewards
            else self.env.default_reward
        )

        if illegal_action:
            reward = self.env.illegal_penalty
        self.update_state()

        # Collision
        if self.angular > 180:
            reward -= 10
        if math.atan2(self.dy, self.dx) == math.atan2(
            self.goal_position[1] - self.y, self.goal_position[0] - self.x
        ):
            reward += 10

        if self.x < 0 or self.x > self.env.dimensions[0] or self.y < 0 or self.y > self.env.dimensions[1]:
            reward = -100
            done = True

        if self.min_dist() < self.length / 1.8:
            reward = -100
            done = True
        if self.min_dist2() == 0:
            reward = -100
            done = True
        elif self.min_dist2() == 0.5:
            reward = -10

        # Goal
        if dist_to_goal < self.length / 1.5:
            self.arrive += 1
            reward = 100 if self.speed > 0 else 50
            done = True

        self.t += 1
        # if self.t % 50 == 0:
        #     self.obstacle = self.update_obstacle()

        return (
            np.array(self.state),
            float(reward),
            done,
            {"dx": self.dx, "dy": self.dy},
        )

    def min_dist(self):
        """求两船（各五个定位点）两点之间位置最小的"""
        dist = []
        X = [
            self.x - 1.2 * self.width * math.cos(math.radians(self.direction + 45)),
            self.x - 1.2 * self.width * math.cos(math.radians(self.direction - 45)),
            self.x + 1 * self.width * math.cos(math.radians(self.direction + 45)),
            self.x + 1 * self.width * math.cos(math.radians(self.direction - 45)),
        ]
        Y = [
            self.y + 1 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.y + 1 * self.width * -math.sin(math.radians(self.direction - 45)),
            self.y - 1.2 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.y - 1.2 * self.width * -math.sin(math.radians(self.direction - 45)),
        ]
        OX = [
            self.other_ship.x
            - 1.2 * self.width * math.cos(math.radians(self.direction + 45)),
            self.other_ship.x
            - 1.2 * self.width * math.cos(math.radians(self.direction - 45)),
            self.other_ship.x
            + 1 * self.width * math.cos(math.radians(self.direction + 45)),
            self.other_ship.x
            + 1 * self.width * math.cos(math.radians(self.direction - 45)),
        ]
        OY = [
            self.other_ship.y
            + 1 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.other_ship.y
            + 1 * self.width * -math.sin(math.radians(self.direction - 45)),
            self.other_ship.y
            - 1.2 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.other_ship.y
            - 1.2 * self.width * -math.sin(math.radians(self.direction - 45)),
        ]
        for x, y in zip(X, Y):
            for ox, oy in zip(OX, OY):
                dist.append(math.hypot(x - ox, y - oy))
        return min(dist)

    def make_contours(self, contours_list):
        if len(contours_list) < 3:
            raise ValueError(
                "The number of contour points \
                                      must be at least three"
            )
        return np.expand_dims(np.array(contours_list), axis=-2)

    def min_dist2(self):
        dist1, dist2 = [], []
        contours_lists = []
        contours = []
        X = [
            self.x - 1.2 * self.width * math.cos(math.radians(self.direction + 45)),
            self.x - 1.2 * self.width * math.cos(math.radians(self.direction - 45)),
            self.x + 1 * self.width * math.cos(math.radians(self.direction + 45)),
            self.x + 1 * self.width * math.cos(math.radians(self.direction - 45)),
        ]
        Y = [
            self.y + 1 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.y + 1 * self.width * -math.sin(math.radians(self.direction - 45)),
            self.y - 1.2 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.y - 1.2 * self.width * -math.sin(math.radians(self.direction - 45)),
        ]

        for i in range(0, len(UAV.oxy), 2):
            contours_list = (
                (UAV.oxy[i], UAV.oxy[i + 1]),
                (UAV.oxy[i] + MAX_OBSTACLE_LEN, UAV.oxy[i + 1]),
                (UAV.oxy[i] + MAX_OBSTACLE_LEN, UAV.oxy[i + 1] + MAX_OBSTACLE_WID),
                (UAV.oxy[i], UAV.oxy[i + 1] + MAX_OBSTACLE_WID),
            )
            contours_i = self.make_contours(contours_list)
            contours_lists.append(contours_list)
            contours.append(contours_i)

        for j in contours:
            dist2.append(cv2.pointPolygonTest(j, (self.x, self.y), True))
            for k in zip(X, Y):
                dist1.append(cv2.pointPolygonTest(j, k, True))
        try:
            if max(dist1) < 0:
                if max(dist2) < 0:
                    if max(dist2) <= -self.length // 1.5:
                        return 0.5
                    else:
                        return 1
                else:
                    return 0
            else:
                return 0
        except:
            return 1

        # if max(dist2) < 0:
        #     if max(dist2) <= -self.length // 1.5:
        #         return 0.5
        #     else:
        #         return 1
        # else:
        #     return 0

    def draw1(self, screen):
        if len(self.tracex) != 0 and len(self.tracey) != 0:
            for tx, ty in zip(self.tracex, self.tracey):
                pg.draw.circle(screen, (0, 0, 255), (int(tx), int(ty)), 1)
        plane_image = pg.transform.rotate(uav, self.direction)
        b_img_w, b_img_h = (plane_image.get_width(), plane_image.get_height())
        pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)
        screen.blit(plane_image, imagerect.move(pivot))
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 2)

        """机头点"""
        # pg.draw.circle(screen, (255, 0, 0), (
        #     self.x + 0.3 * self.ship_length * math.cos(
        #         math.radians(self.direction)),
        #     self.y + 0.3 * self.ship_length * -math.sin(
        #         math.radians(self.direction))),
        #                2)
        """机翼点,前"""
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (
                self.x + 1 * self.width * math.cos(math.radians(self.direction + 45)),
                self.y + 1 * self.width * -math.sin(math.radians(self.direction + 45)),
            ),
            2,
        )
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (
                self.x + 1 * self.width * math.cos(math.radians(self.direction - 45)),
                self.y + 1 * self.width * -math.sin(math.radians(self.direction - 45)),
            ),
            2,
        )

        """机翼点,后"""
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (
                self.x - 1.2 * self.width * math.cos(math.radians(self.direction + 45)),
                self.y
                - 1.2 * self.width * -math.sin(math.radians(self.direction + 45)),
            ),
            2,
        )
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (
                self.x - 1.2 * self.width * math.cos(math.radians(self.direction - 45)),
                self.y
                - 1.2 * self.width * -math.sin(math.radians(self.direction - 45)),
            ),
            2,
        )
        pg.draw.circle(
            screen, (0, 0, 255), (int(self.x), int(self.y)), self.length // 1.5, 1
        )  # 画圈
        pg.draw.circle(
            screen, (0, 0, 255), (int(self.x), int(self.y)), self.length * 2.5, 1
        )  # 画圈，动态障碍出现圈
        pg.draw.circle(
            screen,
            (0, 255, 0),
            (int(self.goal_position[0]), int(self.goal_position[1])),
            self.length // 1.5,
            2,
        )  # 目标绿圈
        other_ship_dist = math.hypot(
            self.x - self.other_ship.x, self.y - self.other_ship.y
        )
        if other_ship_dist < self.length / 3:
            screen.blit(excl_mark, (self.x, self.y))
        for i in range(len(self.obstacle)):
            pg.draw.lines(
                screen,
                (0, 0, 0),
                False,
                [
                    (self.obstacle[i][0], self.obstacle[i][1]),
                    (self.obstacle[i][0] + MAX_OBSTACLE_LEN, self.obstacle[i][1]),
                    (
                        self.obstacle[i][0] + MAX_OBSTACLE_LEN,
                        self.obstacle[i][1] + MAX_OBSTACLE_WID,
                    ),
                    (self.obstacle[i][0], self.obstacle[i][1] + MAX_OBSTACLE_WID),
                    (self.obstacle[i][0], self.obstacle[i][1]),
                ],
                2,
            )

    def draw2(self, screen: pg.Surface):
        plane_image = pg.transform.rotate(uav, self.direction)
        b_img_w, b_img_h = (plane_image.get_width(), plane_image.get_height())
        pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)  # pivot
        screen.blit(plane_image, imagerect.move(pivot))
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 2)
        for ii in range(0, len(UAV.oxy), 2):
            pg.draw.lines(
                screen,
                (0, 0, 0),
                False,
                [
                    (UAV.oxy[ii], UAV.oxy[ii + 1]),
                    (UAV.oxy[ii] + MAX_OBSTACLE_LEN, UAV.oxy[ii + 1]),
                    (
                        UAV.oxy[ii] + MAX_OBSTACLE_LEN,
                        UAV.oxy[ii + 1] + MAX_OBSTACLE_WID,
                    ),
                    (UAV.oxy[ii], UAV.oxy[ii + 1] + MAX_OBSTACLE_WID),
                    (UAV.oxy[ii], UAV.oxy[ii + 1]),
                ],
                2,
            )

    def reset(self, position, speed, direction, goal_position):
        self.x, self.y = position
        self.speed = speed
        self.direction = direction
        self.goal_position = goal_position


class UAVEnvironment(gym.Env):
    def __init__(
        self,
        dimensions: "tuple[int, int]",
        intermediate_rewards: bool = False,
        multi_agent: bool = True,
        default_reward: float = -1,
        illegal_penalty: float = -2,
    ):
        super().__init__()

        # Environmental parameters
        self.dimensions = dimensions

        # RL parameters
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Sequence(gym.spaces.MultiDiscrete([3, 5]))

        self.intermediate_rewards = intermediate_rewards

        # Agents
        self.uav1 = UAV((dimensions[0] // 2, dimensions[1]), self, None)
        self.uav2 = UAV((dimensions[0], 0), self, self.uav1)
        self.uav1.other_ship = self.uav2
        # map A
        self.grid_map = []
        # Misc
        self.screen = None
        self.multi_agent = multi_agent
        self.default_reward = default_reward
        self.illegal_penalty = illegal_penalty
        self.reset()
        self.seed()

    def step(self, action: "list[tuple[int, int]]"):
        for wpt in action:
            self.uav2.step(UAV.CRUISE)
            obs, reward, done, info = self.uav1.step(wpt)
        return obs, reward, done, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_number_in_complex_ranges(self, num):
        # 这里是复杂的逻辑判断
        ranges = [
            (
                (self.uav1.goal_position[0] - (self.uav1.length // 3)),
                (self.uav1.goal_position[0] + (self.uav1.length // 3)),
            ),
            (
                (self.uav1.goal_position[1] - (self.uav1.length // 3)),
                (self.uav1.goal_position[1] + (self.uav1.length // 3)),
            ),
        ]  # 假设有更多的范围
        return any(start < num < end for start, end in ranges)

    def out_of_compliance(self, numbers):
        # 这里只负责遍历列表并调用辅助函数
        return any(self.is_number_in_complex_ranges(num) for num in numbers)

    def reset(self, *, seed=None, options=None):
        uav1_init_x = self.dimensions[0] // 2
        uav1_init_y = self.dimensions[1]
        uav_init_v = np.random.uniform(0, UAV.max_speed)
        uav1_init_angle = 90

        uav2_init_x = np.random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav2_init_y = np.random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav2_init_v = np.random.uniform(0, UAV.max_speed) if self.multi_agent else 0
        uav2_init_angle = np.random.uniform(0, 360)

        self.uav1.reset(
            (uav1_init_x, uav1_init_y),
            uav_init_v,
            uav1_init_angle,
            self.uav1.goal_position,
        )

        self.uav2.reset(
            (
                uav2_init_x,
                uav2_init_y,
            ),
            uav2_init_v,
            uav2_init_angle,
            (
                np.random.uniform(self.dimensions[1] * 0.2, self.dimensions[1] * 0.8),
                np.random.uniform(self.dimensions[1] * 0.2, self.dimensions[1] * 0.8),
            ),
        )
        UAV.oxy = [random.randint(0, 700) for _ in range(NUM_OBSTACLES * 2)]
        while self.out_of_compliance(UAV.oxy):
            UAV.oxy = [random.randint(0, 700) for _ in range(NUM_OBSTACLES * 2)]
        self.grid_map = self.map_for_astar()
        self.uav1.update_state()
        self.uav1.tracex = []
        self.uav1.tracey = []
        self.uav2.update_state()

        return np.array(self.uav1.state), {}

    def map_for_astar(self):
        grid_map = [
            [0 for _ in range(self.dimensions[0])] for _ in range(self.dimensions[0])
        ]
        for i in range(0, NUM_OBSTACLES * 2, 2):
            for j in range(UAV.oxy[i], UAV.oxy[i] + MAX_OBSTACLE_LEN + 1):
                for k in range(UAV.oxy[i + 1], UAV.oxy[i + 1] + MAX_OBSTACLE_WID + 1):
                    try:
                        grid_map[k][j] = 1
                    except:
                        pass
        grid_map[self.uav1.goal_position[1]][self.uav1.goal_position[0]] = 2

        return grid_map

    def render(self):
        if self.screen is None:
            self._init_pygame()

        self.draw()
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()

    def draw(self):
        # water
        pg.draw.rect(
            self.screen,
            (66, 84, 155),
            pg.Rect(0, 0, self.dimensions[0], self.dimensions[1]),
        )
        self.screen.blit(background, (0, 0))
        self.uav1.draw1(self.screen)
        self.uav2.draw2(self.screen)

        text_surface, _ = font.render(
            "UAV1: spd={:.2f}, angvel={:.2f}".format(
                self.uav1.speed, self.uav1.angular_velocity
            ),
            (0, 0, 0),
        )
        pg.draw.line(
            self.screen,
            (255, 0, 0),
            (16, 48),
            (
                16 + 16 * math.cos(math.radians(100 * self.uav1.angular_velocity)),
                48 + 16 * math.sin(math.radians(100 * self.uav1.angular_velocity)),
            ),
        )
        self.screen.blit(text_surface, (16, 16))

    def close(self):
        if self.screen is not None:
            pg.quit()
            self.screen = None

    def _init_pygame(self):
        pg.init()
        self.screen = pg.display.set_mode(self.dimensions)
        pg.display.set_caption("UAV Environment")


"""A*"""


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


class AStar:
    def __init__(self, grid, start, goal, other_b):
        self.grid = grid
        self.start = Node(None, start)
        self.goal = goal
        self.open_set = []
        self.closed_set = set()
        self.other_b = other_b

    def heuristic(self, a, b):
        # 使用曼哈顿距离作为启发式函数
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node):
        neighbors = []
        row, col = node.position
        directions = [
            (-1, 1),
            (0, 1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (1, -1),
        ]  # 上下左右
        directions = [
            (5 * x, 5 * y) for x, y in directions
        ]  # 防止地图栅格过多，计算时间变长，但间隔过大可能导致跨过障碍
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            # 增加动态障碍other_b
            mapp = self.grid
            # try:
            #     mapp[int(self.other_b.x)][int(self.other_b.y)] = 1
            # except:
            #     pass
            # 检查新坐标是否在网格范围内
            if 0 <= new_row < len(mapp) and 0 <= new_col < len(mapp[0]):
                # 检查新坐标是否不是障碍物
                new_row = int(new_row)
                new_col = int(new_col)
                if (
                    mapp[new_row][new_col] != 1
                    and (new_row, new_col) != self.start.position
                ):
                    new_node = Node(node, (new_row, new_col))
                    neighbors.append(new_node)

        return neighbors

    def a_star_search(self):
        self.open_set.append(self.start)

        while self.open_set:
            # 选择f值最小的节点
            current = min(self.open_set, key=lambda node: node.f)
            self.open_set.remove(current)
            self.closed_set.add(current)

            if current.position == self.goal:
                return self.reconstruct_path(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = current.g + 1

                if tentative_g_score < neighbor.g or neighbor not in self.open_set:
                    neighbor.g = tentative_g_score
                    neighbor.h = self.heuristic(neighbor.position, self.goal)
                    neighbor.f = neighbor.g + neighbor.h

                    if neighbor not in self.open_set:
                        self.open_set.append(neighbor)
                        # return neighbor.position

        return None  # 如果没有找到路径

    def reconstruct_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # 反转列表以得到正确的路径顺序


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.render:
        clock = pg.time.Clock()

    dim = (args.width, args.height)
    env = UAVEnvironment(dim, intermediate_rewards=True, multi_agent=True)

    for _ in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            if args.render:
                env.render()
                clock.tick_busy_loop(60)

    env.close()
