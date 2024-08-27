from __future__ import annotations

import math
import os
import random
import time

import weakref
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import pygame.freetype as pgft
from gymnasium.utils import seeding
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit

root = os.path.dirname(os.path.abspath(__file__))
uav = pg.image.load(os.path.join(root, "res/uav2.jpg"))
uav = pg.transform.rotozoom(uav, 0, 0.22)

background = pg.image.load(os.path.join(root, "res/sky.png"))
background = pg.transform.rotozoom(background, 0, 2)
imagerect = uav.get_rect()

missile = pg.image.load(os.path.join(root, "res/missile.png"))
missile = pg.transform.rotozoom(missile, 0, 0.35)
imagerect_missile = missile.get_rect()

pgft.init()
font = pgft.SysFont("", 20)
excl_mark, _ = font.render("!!!", (255, 0, 0))

STATE_SIZE = 11


class Obstacle:

    def __init__(self, fixed_number, dynamic_number):
        self.fixed_number = fixed_number
        self.dynamic_number = dynamic_number
        self.fixed_position = []
        self.fixed_wid = []
        self.fixed_len = []
        self.dynamic_position = []
        self.dynamic_direction = []
        self.dynamic_speed = []
        self.dynamic_angular_velocity = []
        self.dynamic_mode = []

        self._fixed = []
        self._dynamic = []

    def reset(self):
        self.init_fixed()
        self.init_dynamic()

    def init_fixed(self):
        self.fixed_position.clear()
        self.fixed_len.clear()
        self.fixed_wid.clear()
        self._fixed.clear()

        if self.fixed_number != 0:
            # 固定障碍物
            for i in range(self.fixed_number):
                self.fixed_position.append(
                    (random.randint(0, 750), random.randint(0, 750))
                )
                self.fixed_len.append(int(np.random.uniform(10, 80)))
                self.fixed_wid.append(int(np.random.uniform(10, 80)))

    def init_dynamic(self):
        self.dynamic_position.clear()
        self.dynamic_direction.clear()
        self.dynamic_speed.clear()
        self.dynamic_angular_velocity.clear()
        self.dynamic_mode.clear()
        self._dynamic.clear()

        if self.dynamic_number != 0:
            for _ in range(self.dynamic_number):
                self.dynamic_position.append(
                    (random.randint(50, 750), random.randint(50, 750))
                )
                self.dynamic_direction.append(random.randint(0, 360))
                self.dynamic_speed.append(random.randint(10, 100) * 0.01)
                self.dynamic_angular_velocity.append(random.randint(0, 100) * 0.01)
                self.dynamic_mode.append(random.randint(0, 1))  # 0直线行驶，1绕圈

    @property
    def fixed_obstacle(self):
        """所有障碍点位置"""
        if not self._fixed:
            for i in range(self.fixed_number):
                for j in range(
                    self.fixed_position[i][0],
                    self.fixed_position[i][0] + self.fixed_wid[i],
                ):
                    for k in range(
                        self.fixed_position[i][1],
                        self.fixed_position[i][1] + self.fixed_len[i],
                    ):
                        self._fixed.append((j, k))
        return self._fixed

    @property
    def dynamic_obstacle(self):
        if not self._dynamic:
            for i in range(self.dynamic_number):
                for j in range(
                    self.dynamic_position[i][0],
                    self.dynamic_position[i][0] + DynamicObstacle.width * 2,
                ):
                    for k in range(
                        self.dynamic_position[i][1],
                        self.dynamic_position[i][1] + DynamicObstacle.length * 2,
                    ):
                        self._dynamic.append((j, k))

        return self._dynamic


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


# 算两向量夹角
def calculate_angle_between_vectors(x, y, a, b):
    """
    计算两个向量(x, y)和(a, b)之间的夹角（以度为单位）。
    """
    # 计算点积
    dot_product = x * a + y * b

    # 计算两个向量的模长
    norm_v1 = math.sqrt(x**2 + y**2)
    norm_v2 = math.sqrt(a**2 + b**2)

    # 避免除以零的错误
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # 或者可以抛出一个异常，表示向量之一是零向量

    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # 使用acos计算夹角（以弧度为单位），然后转换为度
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# 贝塞尔曲线函数
def quadratic_bezier(p0, p1, p2, t):
    """
    计算并返回二次贝塞尔曲线在参数t下的点。
    p0, p1, p2 是控制点(numpy数组), t是参数(0 <= t <= 1)
    """
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


# 附体坐标系转大地坐标系
def transform_coordinates(a, b, direx):
    """
    Transform coordinates (a, b) in the body frame to the general frame.

    Parameters:
    a, b (float): Coordinates in the body frame.
    direx (float): Angle between the body frame and the general frame (radians).

    Returns:
    tuple: Transformed coordinates in the general frame.
    """
    # Rotation matrix
    cos_direx = math.cos(direx)
    sin_direx = math.sin(direx)
    R = [[cos_direx, -sin_direx], [sin_direx, cos_direx]]

    # Transform coordinates
    general_x = a * R[0][0] + b * R[0][1]
    general_y = a * R[1][0] + b * R[1][1]

    return (general_x, general_y)


class UAV:
    max_speed = 1
    min_speed = 0.1
    max_ang_vel = 0.3
    length = 40
    width = 40

    def __init__(
        self,
        pos: "tuple[int, int]",
        env: "UAVEnvironment",
        initial_speed: float = 0,
        initial_direction: float = 0,
        initial_angular_velocity: float = 0,
        goal_position: "tuple[int, int]" = (
            random.randint(100, 600),
            random.randint(100, 600),
        ),
    ):
        self.env: "UAVEnvironment" = weakref.proxy(env)

        # Internal state variables
        self.x, self.y = pos
        self.dx = 0
        self.dy = 0
        self.speed = initial_speed
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity
        self.prev_dist = 0
        self.dist_to_goal = 0

        # Episode dependent variables
        self.goal_position = goal_position
        self.show_circles = False
        self.arrive = 0

        self.tracex = []
        self.tracey = []
        self.angular = 0
        self.angular_to_goal = 0

    @property
    def position(self):
        return (self.x, self.y)

    def update_state(self):  # state加主飞机前面的随即障碍
        g_nowx, g_nowy = self.goal_position[0] - self.x, self.goal_position[1] - self.y
        prev_to_goal = self.dist_to_goal
        self.dist_to_goal = math.hypot(
            self.x - self.goal_position[0], self.y - self.goal_position[1]
        )
        self.prev_dist = self.dist_to_goal - prev_to_goal
        self.angular_to_goal = calculate_angle_between_vectors(
            self.goal_position[0], self.goal_position[1], self.x, self.y
        )

        goal_offset = angular_offset(self, self.goal_position)

        c_offset = []
        dist_dynamic = []
        for d_obstacle in self.env.d_obstacles:
            c_offset.append(angular_offset(self, d_obstacle.position))
            dist_dynamic.append(
                math.hypot(self.x - d_obstacle.x, self.y - d_obstacle.y)
            )
        collision_offset = min(c_offset, default=0)
        dist_to_dynamic = min(dist_dynamic, default=0)

        dist_to_fixed = 0
        dist_l = []
        for i in range(self.env.obstacles.fixed_number):
            dist_l.append(
                math.hypot(
                    self.x - self.env.obstacles.fixed_position[i][0],
                    self.y - self.env.obstacles.fixed_position[i][1],
                )
            )
        dist_to_fixed = min(dist_l, default=0)

        self.state = [
            g_nowx,
            g_nowy,
            self.dist_to_goal,
            self.prev_dist,
            self.speed / UAV.max_speed,
            self.angular_velocity / UAV.max_ang_vel,
            self.angular_to_goal,
            goal_offset,
            collision_offset,
            dist_to_dynamic,
            dist_to_fixed,
        ]

    def step(self, action: "tuple[int, int]", astar: bool = False):
        done = False
        illegal_action = False
        info = {"done": "not_done"}
        speed_action, turn_action = action

        # Define the conversion factors
        increments_per_grid = 1
        speed_increment = 1
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
        info["dist_to_goal"] = dist_to_goal

        reward = (
            -float(self.prev_dist)
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

        if (
            self.x < 0
            or self.x > self.env.dimensions[0]
            or self.y < 0
            or self.y > self.env.dimensions[1]
        ):
            reward = -100
            done = True
            info["done"] = "out_of_bounds"

        obstacles = self.env.obstacles
        if obstacles.dynamic_number != 0:
            if self.dynamic_obstacle_collision() < self.length + DynamicObstacle.length:
                reward = -100
                done = True
                info["done"] = "collide_with_dynamic"
        if self.fixed_obstacle_collision() == 0:
            reward = -100
            done = True
            info["done"] = "collide_with_obstacle"
        elif self.fixed_obstacle_collision() == 0.5:
            reward = -10

        # Goal
        if dist_to_goal < self.length:
            self.arrive += 1
            reward = 100 if self.speed > 0 else 50
            done = True
            info["done"] = "reach_goal"

        return (
            np.array(self.state),
            float(reward),
            done,
            info,
        )

    def dynamic_obstacle_collision(self):
        """求两无人机（各五个定位点）两点之间位置最小的"""
        ll = []
        for d_obstacle in self.env.d_obstacles:
            ll.append(math.hypot(self.x - d_obstacle.x, self.y - d_obstacle.y))
        return min(ll, default=math.inf)

    def make_contours(self, contours_list):
        if len(contours_list) < 3:
            raise ValueError(
                "The number of contour points \
                                      must be at least three"
            )
        return np.expand_dims(np.array(contours_list), axis=-2)

    def fixed_obstacle_collision(self):
        dist1, dist2 = [], []
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

        obstacles = self.env.obstacles
        for i in range(obstacles.fixed_number):
            contours_list = (
                (obstacles.fixed_position[i][0], obstacles.fixed_position[i][1]),
                (
                    obstacles.fixed_position[i][0] + obstacles.fixed_wid[i],
                    obstacles.fixed_position[i][1],
                ),
                (
                    obstacles.fixed_position[i][0] + obstacles.fixed_wid[i],
                    obstacles.fixed_position[i][1] + obstacles.fixed_len[i],
                ),
                (
                    obstacles.fixed_position[i][0],
                    obstacles.fixed_position[i][1] + obstacles.fixed_len[i],
                ),
            )
            contours_i = self.make_contours(contours_list)
            contours.append(contours_i)

        for j in contours:
            dist2.append(cv2.pointPolygonTest(j, (self.x, self.y), True))
            for k in zip(X, Y):
                dist1.append(cv2.pointPolygonTest(j, k, True))
        try:
            if max(dist1) < 0:
                if max(dist2) < -self.length * 1.2:
                    if max(dist2) <= -self.length * 3.5:
                        return 0.5
                    else:
                        return 1
                else:
                    return 0
            else:
                return 0
        except:
            return 1

    def draw(self, screen: pg.Surface):
        if len(self.tracex) != 0 and len(self.tracey) != 0:
            for tx, ty in zip(self.tracex, self.tracey):
                pg.draw.circle(screen, (0, 0, 255), (int(tx), int(ty)), 1)
        plane_image = pg.transform.rotate(uav, self.direction)
        b_img_w, b_img_h = (plane_image.get_width(), plane_image.get_height())
        pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)
        screen.blit(plane_image, imagerect.move(pivot))
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 2)

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
            screen, (0, 0, 255), (int(self.x), int(self.y)), self.length * 1.1, 1
        )  # 画圈
        pg.draw.circle(
            screen, (0, 0, 255), (int(self.x), int(self.y)), self.length * 3.5, 1
        )  # 画圈，动态障碍出现圈，也是警戒范围？
        pg.draw.circle(
            screen,
            (0, 255, 0),
            (int(self.goal_position[0]), int(self.goal_position[1])),
            self.length,
            2,
        )  # 目标绿圈

        obstacles = self.env.obstacles
        for d_obstacle in self.env.d_obstacles:
            other_uav_dist = math.hypot(self.x - d_obstacle.x, self.y - d_obstacle.y)
            if other_uav_dist < self.length * 2:
                screen.blit(excl_mark, (self.x, self.y))

        for ii in range(obstacles.fixed_number):
            pg.draw.lines(
                screen,
                (0, 0, 0),
                False,
                [
                    (obstacles.fixed_position[ii][0], obstacles.fixed_position[ii][1]),
                    (
                        obstacles.fixed_position[ii][0] + obstacles.fixed_wid[ii],
                        obstacles.fixed_position[ii][1],
                    ),
                    (
                        obstacles.fixed_position[ii][0] + obstacles.fixed_wid[ii],
                        obstacles.fixed_position[ii][1] + obstacles.fixed_len[ii],
                    ),
                    (
                        obstacles.fixed_position[ii][0],
                        obstacles.fixed_position[ii][1] + obstacles.fixed_len[ii],
                    ),
                    (obstacles.fixed_position[ii][0], obstacles.fixed_position[ii][1]),
                ],
                2,
            )

    def reset(self, position, speed, direction, goal_position):
        self.x, self.y = position
        self.speed = speed
        self.direction = direction
        self.goal_position = goal_position


class DynamicObstacle:
    length = 30
    width = 10

    def __init__(
        self,
        pos: "tuple[int, int]",
        initial_speed: float = 0,
        initial_direction: float = 0,
        initial_angular_velocity: float = 0,
        mode: int = 0,
    ):
        # Internal state variables
        self.x, self.y = pos
        self.dx = 0
        self.dy = 0
        self.speed = initial_speed
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity

        # Episode dependent variables
        self.mode = mode
        self.angular = 0

    @property
    def position(self):
        return (self.x, self.y)

    def step(self):
        if self.mode == 1:
            self.dx = self.speed * math.cos(math.radians(self.direction))
            self.dy = self.speed * -math.sin(math.radians(self.direction))
            self.x += self.dx
            self.y += self.dy
            self.direction = (self.direction + self.angular_velocity) % 360
        else:
            self.speed = self.speed
            self.angular = 0
            self.angular_velocity = 0
            self.dx = self.speed * math.cos(math.radians(self.direction))
            self.dy = self.speed * -math.sin(math.radians(self.direction))
            self.x += self.dx
            self.y += self.dy
            self.direction = (self.direction + self.angular_velocity) % 360

    def draw(self, screen: pg.Surface):
        plane_image = pg.transform.rotate(missile, self.direction)
        b_img_w, b_img_h = (plane_image.get_width(), plane_image.get_height())
        pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)  # pivot
        screen.blit(plane_image, imagerect_missile.move(pivot))
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 2)

        pg.draw.circle(
            screen,
            (255, 0, 0),
            (
                self.x + 1 * self.length * math.cos(math.radians(self.direction)),
                self.y + 1 * self.length * -math.sin(math.radians(self.direction)),
            ),
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
        fixed_obstacles: int = 0,
        dynamic_obstacles: int = 0,
        intermediate_rewards: bool = False,
        multi_agent: bool = True,
        default_reward: float = -1,
        illegal_penalty: float = -2,
    ):
        super().__init__()

        # Environmental parameters
        self.dimensions = dimensions
        self.obstacles = Obstacle(fixed_obstacles, dynamic_obstacles)

        # RL parameters
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32
        )

        # Use gym.spaces.Sequence() for multi waypoint
        self.action_space = gym.spaces.MultiDiscrete([3, 5])

        self.intermediate_rewards = intermediate_rewards

        # Agents
        self.uav = UAV((dimensions[0] // 2, dimensions[1]), self, None)
        self.d_obstacles: "list[DynamicObstacle]" = []

        # map A
        self.grid_map = []
        # Misc
        self.screen = None
        self.multi_agent = multi_agent
        self.default_reward = default_reward
        self.illegal_penalty = illegal_penalty
        self.reset()
        self.seed()

    def step(self, action: "tuple[int, int]", astar: bool = False):
        if self.obstacles.dynamic_number != 0:
            for d_obstacle in self.d_obstacles:
                d_obstacle.step()
        obs, reward, done, info = self.uav.step(action, astar)
        return obs, reward, done, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        self.uav.goal_position = (random.randint(100, 600), random.randint(100, 600))
        uav_init_x = np.random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav_init_y = np.random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )

        uav_init_v = 0
        uav1_init_angle = 0
        self.uav.reset(
            (uav_init_x, uav_init_y),
            uav_init_v,
            uav1_init_angle,
            self.uav.goal_position,
        )

        self.obstacles.reset()
        self.d_obstacles.clear()

        # 障碍物与目标点重合或者距离过近, 重新生成
        while (uav_init_x, uav_init_y) in self.obstacles.fixed_obstacle and (
            math.hypot(
                uav_init_x - self.uav.goal_position[0],
                uav_init_y - self.uav.goal_position[1],
            )
            < UAV.width * 2
        ):
            self.obstacles.init_fixed()

        if self.obstacles.dynamic_number != 0:
            for i in range(self.obstacles.dynamic_number):
                init_x = self.obstacles.dynamic_position[i][0]
                init_y = self.obstacles.dynamic_position[i][1]
                init_v = self.obstacles.dynamic_speed[i]
                init_angle = self.obstacles.dynamic_direction[i]
                d_obstacle = DynamicObstacle(self.obstacles.dynamic_obstacle[i])

                d_obstacle.reset(
                    (init_x, init_y),
                    init_v,
                    init_angle,
                    (0, 0),
                )
                self.d_obstacles.append(d_obstacle)

        self.grid_map = self.map_for_astar(int(uav_init_x), int(uav_init_y))
        self.uav.update_state()
        self.uav.tracex = []
        self.uav.tracey = []

        return np.array(self.uav.state), {}

    def map_for_astar(self, y, x):
        grid_map = [
            [0 for _ in range(self.dimensions[0])] for _ in range(self.dimensions[0])
        ]
        for i in range(self.obstacles.fixed_number):
            for j in range(
                self.obstacles.fixed_position[i][0] - UAV.width - 10,
                self.obstacles.fixed_position[i][0]
                + self.obstacles.fixed_wid[i]
                + UAV.width
                + 11,
            ):
                for k in range(
                    self.obstacles.fixed_position[i][1] - UAV.length - 10,
                    self.obstacles.fixed_position[i][1]
                    + self.obstacles.fixed_len[i]
                    + UAV.length
                    + 11,
                ):
                    try:
                        grid_map[k][j] = 1
                    except:
                        pass
        grid_map[self.uav.goal_position[1]][self.uav.goal_position[0]] = 2
        grid_map[y][x] = 3
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
        self.uav.draw(self.screen)
        if self.obstacles.dynamic_number != 0:
            for d_obstacle in self.d_obstacles:
                d_obstacle.draw(self.screen)

        text_surface, _ = font.render(
            "UAV1: spd={:.2f}, angvel={:.2f}".format(
                self.uav.speed, self.uav.angular_velocity
            ),
            (0, 0, 0),
        )
        pg.draw.line(
            self.screen,
            (255, 0, 0),
            (16, 48),
            (
                16 + 16 * math.cos(math.radians(100 * self.uav.angular_velocity)),
                48 + 16 * math.sin(math.radians(100 * self.uav.angular_velocity)),
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
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = Node(None, start)
        self.goal = goal
        self.open_set: "list[Node]" = []
        self.closed_set = set()
        self.other_b = 0  # A*未加入动态障碍信息

    def heuristic(self, a, b):
        # 使用曼哈顿距离作为启发式函数
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node: Node):
        neighbors = []
        directions = []
        row, col = node.position
        for i in range(-2, 3):
            for j in range(-2, 4):
                directions.append((i, j))

        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            # 增加动态障碍other_b
            mapp = self.grid

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
        steps = 1000
        while self.open_set and steps > 0:
            steps -= 1
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

        return None  # 如果没有找到路径

    def reconstruct_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # 反转列表以得到正确的路径顺序


def astar_test(env: UAVEnvironment, episodes: int = 3, render: bool = False):
    if render:
        clock = pg.time.Clock()

    for i in range(episodes):
        env.reset()
        done = False
        plt.cla()
        astar = AStar(
            env.grid_map,
            (int(env.uav.y), int(env.uav.x)),
            (env.uav.goal_position[1], env.uav.goal_position[0]),
        )
        print(
            "起点：",
            (int(env.uav.x), int(env.uav.y)),
            "终点：",
            (env.uav.goal_position[0], env.uav.goal_position[1]),
        )
        start_time = time.time()
        path = astar.a_star_search()
        end_time = time.time()

        if path is None:
            print(f"Episode {i}, No path found")
            continue
        else:
            print(path)

        for p in path:
            env.grid_map[int(p[0]) - 1][int(p[1]) - 1] = 3

        print("AStar runtime:", end_time - start_time)
        plt.imshow(env.grid_map, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.show()
        follow_position = 0

        while not done:
            plt.cla()
            if (follow_position + 2) != len(path):
                delta_x1 = path[follow_position + 1][0] - path[follow_position][0]
                delta_y1 = path[follow_position + 1][1] - path[follow_position][1]
                delta_x2 = path[follow_position + 2][0] - path[follow_position + 1][0]
                delta_y2 = path[follow_position + 2][1] - path[follow_position + 1][1]
            else:
                delta_x1, delta_y1 = 0, 0
                delta_x2, delta_y2 = 0, 0

            if (delta_y2 == delta_y1) and (delta_x1 == delta_x2):  # 拐点判定
                curr_po = (env.uav.x, env.uav.y)  # 当前点位置
                # 跟踪点位置（matplot与pygame坐标轴xy相反）
                next_po = path[follow_position]
                # 与跟踪点之间向量差follow_vector：（turn, speed）
                follow_vector = [next_po[1] - curr_po[0], next_po[0] - curr_po[1]]

                action = [(follow_vector[1], follow_vector[0])]
                _, _, done, _, info = env.step(action, astar=True)
                if render:
                    env.render()
                    clock.tick_busy_loop(60)
                follow_position += 1
            else:
                p0 = np.array((path[follow_position][1], path[follow_position][0]))
                p1 = np.array(
                    (path[follow_position + 1][1], path[follow_position + 1][0])
                )
                p2 = np.array(
                    (path[follow_position + 2][1], path[follow_position + 2][0])
                )
                # 创建一个t的数组，用于生成曲线上的点
                bezier_t = np.linspace(0, 1, 15)
                # 计算曲线上的点
                bezier_points = np.array(
                    [quadratic_bezier(p0, p1, p2, ti) for ti in bezier_t]
                )

                follow_point = 0
                bezier_len = len(bezier_points[:, 1]) - 1
                for point in range(bezier_len):
                    curr_po = (env.uav.x, env.uav.y)
                    # 跟踪点位置（matplot与pygame坐标轴xy相反）
                    next_po = (
                        bezier_points[:, 1][follow_point],
                        bezier_points[:, 0][follow_point],
                    )
                    # 与跟踪点之间向量差follow_vector：（turn,speed）
                    follow_vector = [
                        next_po[1] - curr_po[0],
                        next_po[0] - curr_po[1],
                    ]

                    action = [(follow_vector[1], follow_vector[0])]
                    _, _, done, _, info = env.step(action, astar=True)
                    if render:
                        env.render()
                        clock.tick_busy_loop(60)
                    follow_point += 1
                follow_position += 2
        else:
            print(f"Episode {i} done due to {info['done']}")

    env.close()


def rl_test(env: UAVEnvironment, episodes: int = 3, render: bool = False):
    if render:
        clock = pg.time.Clock()

    for i in range(episodes):
        env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            _, _, done, _, info = env.step(action)
            if render:
                env.render()
                clock.tick_busy_loop(60)
        else:
            print(f"Episode {i} done due to {info['done']}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--astar", action="store_true")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fixed_obstacle_number", type=int, default=4)
    parser.add_argument("--dynamic_obstacle_number", type=int, default=2)
    args = parser.parse_args()

    dim = (args.width, args.height)
    env = UAVEnvironment(
        dim,
        args.fixed_obstacle_number,
        args.dynamic_obstacle_number,
        intermediate_rewards=True,
        multi_agent=True,
    )

    args.astar = False
    if args.astar:
        astar_test(env, args.episodes, args.render)
    else:
        rl_test(env, args.episodes, args.render)
