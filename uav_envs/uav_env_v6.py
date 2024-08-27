from __future__ import annotations

import math
import os
import queue
import random
import threading
import time
import weakref

import cv2
import gymnasium as gym
# import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import pygame.freetype as pgft
from gymnasium.utils import seeding

from uav_envs.utils.lidar import Point2d, Structure, Lidar
from uav_envs.wrappers.pretext_wrapper import PretextWrapper
from uav_envs.wrappers.raster_wrapper import RasterWrapper

# from scipy.interpolate import splev, splrep
# from scipy.optimize import curve_fit

root = os.path.dirname(os.path.abspath(__file__))
uav = pg.image.load(os.path.join(root, "res/uav2.jpg"))
uav = pg.transform.rotozoom(uav, 0, 0.14)

background = pg.image.load(os.path.join(root, "res/sky.png"))
background = pg.transform.rotozoom(background, 0, 2.2)
imagerect = uav.get_rect()

missile = pg.image.load(os.path.join(root, "res/missile.png"))
missile = pg.transform.rotozoom(missile, 0, 0.35)
imagerect_missile = missile.get_rect()

pgft.init()
font = pgft.SysFont("", 20)
excl_mark, _ = font.render("!!!", (255, 0, 0))

STATE_SIZE = 5
MAX_TIMESTEP = 2_000


class Obstacle:

    def __init__(self, fixed_number: int, occur_number: int, occur_number_max: int, dynamic_number: int):
        self.fixed_number = fixed_number
        self.occur_fixed_number = occur_number
        self.occur_number_max = occur_number_max
        self.dynamic_number = dynamic_number

        self.fixed_position = []
        self.fixed_wid = []
        self.fixed_len = []

        self.occur_fixed_position = []
        self.occur_fixed_angular = []
        self.occur_fixed_wid = []
        self.occur_fixed_len = []

        self.dynamic_position = []
        self.dynamic_direction = []
        self.dynamic_speed = []
        self.dynamic_angular_velocity = []
        self.dynamic_mode = []

        self._fixed = []
        self._occur = []
        self._dynamic = []

    def reset(self, x, y, dire):
        self.init_fixed()
        self.init_occur_fixed(x, y, dire)

    # 初始各障碍位置长短等参数
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

    def init_occur_fixed(self, x, y, dire):
        self.occur_fixed_position.clear()
        self.occur_fixed_angular.clear()
        self.occur_fixed_len.clear()
        self.occur_fixed_wid.clear()
        self._occur.clear()

        if self.occur_fixed_number != 0:
            # 固定障碍物
            for i in range(self.occur_fixed_number):
                self.occur_fixed_len.append(int(np.random.uniform(10, 30)))
                self.occur_fixed_wid.append(int(np.random.uniform(10, 30)))
                self.occur_fixed_angular.append(
                    int(np.random.uniform(dire - 90, dire + 90))
                )
                self.occur_fixed_position.append(
                    (
                        round(
                            x
                            + 6
                            * UAV.length
                            * math.cos(math.radians(self.occur_fixed_angular[-1]))
                        ),
                        round(
                            y
                            + 6
                            * UAV.length
                            * -math.sin(math.radians(self.occur_fixed_angular[-1]))
                        ),
                    )
                )

    # 更新临时机制的固定障碍
    def update_occur_obstacle(self, x, y, dire):
        if len(self.occur_fixed_position) < self.occur_number_max:
            pass
        else:
            self.occur_fixed_len = self.occur_fixed_len[self.occur_fixed_number:]
            self.occur_fixed_wid = self.occur_fixed_wid[self.occur_fixed_number:]
            self.occur_fixed_angular = self.occur_fixed_angular[
                                       self.occur_fixed_number:
                                       ]
            self.occur_fixed_position = self.occur_fixed_position[
                                        self.occur_fixed_number:
                                        ]
        for i in range(self.occur_fixed_number):
            self.occur_fixed_len.append(int(np.random.uniform(10, 30)))
            self.occur_fixed_wid.append(int(np.random.uniform(10, 30)))
            self.occur_fixed_angular.append(
                int(np.random.uniform(dire - 90, dire + 90))
            )
            self.occur_fixed_position.append(
                (
                    round(
                        x
                        + 6
                        * UAV.length
                        * math.cos(math.radians(self.occur_fixed_angular[-1]))
                    ),
                    round(
                        y
                        + 6
                        * UAV.length
                        * -math.sin(math.radians(self.occur_fixed_angular[-1]))
                    ),
                )
            )

    def init_dynamic(self, uav_position, goal_position):
        self.dynamic_position.clear()
        self.dynamic_direction.clear()
        self.dynamic_speed.clear()
        self.dynamic_angular_velocity.clear()
        self.dynamic_mode.clear()
        self._dynamic.clear()

        if self.dynamic_number != 0:
            for _ in range(self.dynamic_number):
                dynamic_x = random.randint(50, 750)
                dynamic_y = random.randint(50, 750)
                self.dynamic_position.append((dynamic_x, dynamic_y))
                try:
                    angular_dynamic_uav = -calculate_angle_between_vectors(1,
                                                                           0,
                                                                           uav_position[0] - dynamic_x,
                                                                           uav_position[1] - dynamic_y)
                    angular_dynamic_goal = -calculate_angle_between_vectors(1,
                                                                            0,
                                                                            goal_position[0] - dynamic_x,
                                                                            goal_position[1] - dynamic_y)
                except:
                    angular_dynamic_goal = 0
                    angular_dynamic_uav = 0
                # print('.............')
                # print(angular_dynamic_uav)
                # print(angular_dynamic_goal)
                if angular_dynamic_goal > angular_dynamic_uav:
                    min_angular = angular_dynamic_uav
                    max_angular = angular_dynamic_goal
                else:

                    min_angular = angular_dynamic_goal
                    max_angular = angular_dynamic_uav
                self.dynamic_direction.append(
                    random.randint(round(min_angular),
                                   round(max_angular)))
                self.dynamic_speed.append(random.randint(10, 20))
                self.dynamic_angular_velocity.append(random.randint(0, 100) * 0.01)
                self.dynamic_mode.append(random.randint(0, 1))  # 0直线行驶，1绕圈

    # 统计各障碍所在位置，方便初始化时不与障碍重叠
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
    def occur_fixed_obstacle(self):
        """所有障碍点位置"""
        if not self._occur:
            for i in range(self.occur_fixed_number):
                for j in range(
                        self.occur_fixed_position[i][0],
                        self.occur_fixed_position[i][0] + self.occur_fixed_wid[i],
                ):
                    for k in range(
                            self.occur_fixed_position[i][1],
                            self.occur_fixed_position[i][1] + self.occur_fixed_len[i],
                    ):
                        self._occur.append((j, k))
        return self._occur

    @property
    def dynamic_obstacle(self):
        if not self._dynamic:
            for i in range(self.dynamic_number):
                for j in range(
                        self.dynamic_position[i][0] - DynamicObstacle.width * 2,
                        self.dynamic_position[i][0] + DynamicObstacle.width * 2,
                ):
                    for k in range(
                            self.dynamic_position[i][1] - DynamicObstacle.length * 2,
                            self.dynamic_position[i][1] + DynamicObstacle.length * 2,
                    ):
                        self._dynamic.append((j, k))

        return self._dynamic

    def get_fixed_structures(self):
        results = []
        for i in range(len(self.fixed_position)):
            results.append(Structure([
                Point2d(self.fixed_position[i][0],
                        self.fixed_position[i][1]),
                Point2d(self.fixed_position[i][0],
                        self.fixed_position[i][1] + self.fixed_len[i]),
                Point2d(self.fixed_position[i][0] + self.fixed_wid[i],
                        self.fixed_position[i][1] + self.fixed_len[i]),
                Point2d(self.fixed_position[i][0] + self.fixed_wid[i],
                        self.fixed_position[i][1]),
            ]))
        return results

    def get_occur_structures(self):
        results = []
        for i in range(len(self.occur_fixed_position)):
            results.append(Structure([
                Point2d(self.occur_fixed_position[i][0],
                        self.occur_fixed_position[i][1]),
                Point2d(self.occur_fixed_position[i][0],
                        self.occur_fixed_position[i][1] + self.occur_fixed_len[i]),
                Point2d(self.occur_fixed_position[i][0] + self.occur_fixed_wid[i],
                        self.occur_fixed_position[i][1] + self.occur_fixed_len[i]),
                Point2d(self.occur_fixed_position[i][0] + self.occur_fixed_wid[i],
                        self.occur_fixed_position[i][1]),
            ]))
        return results

    def get_dynamic_structures(self):
        results = []
        vertex_angle = math.degrees(math.atan2(DynamicObstacle.length, DynamicObstacle.width))
        vertex_angle_comp = 90. - vertex_angle
        for i in range(len(self.dynamic_position)):
            results.append(Structure([
                Point2d(self.dynamic_position[i][0] + DynamicObstacle.length
                        * math.cos(math.radians(0. + self.dynamic_direction[i] + vertex_angle_comp)),
                        self.dynamic_position[i][1] + DynamicObstacle.length
                        * math.cos(math.radians(0. + self.dynamic_direction[i] + vertex_angle_comp))),
                Point2d(self.dynamic_position[i][0] + DynamicObstacle.length
                        * math.cos(math.radians(90. + self.dynamic_direction[i] + vertex_angle)),
                        self.dynamic_position[i][1] + DynamicObstacle.length
                        * math.cos(math.radians(90. + self.dynamic_direction[i] + vertex_angle))),
                Point2d(self.dynamic_position[i][0] + DynamicObstacle.length
                        * math.cos(math.radians(180. + self.dynamic_direction[i] + vertex_angle_comp)),
                        self.dynamic_position[i][1] + DynamicObstacle.length
                        * math.cos(math.radians(180. + self.dynamic_direction[i] + vertex_angle_comp))),
                Point2d(self.dynamic_position[i][0] + DynamicObstacle.length
                        * math.cos(math.radians(270. + self.dynamic_direction[i] + vertex_angle)),
                        self.dynamic_position[i][1] + DynamicObstacle.length
                        * math.cos(math.radians(270. + self.dynamic_direction[i] + vertex_angle))),
            ]))
        return results

    def get_all_structures(self):
        return self.get_fixed_structures() + self.get_occur_structures()


def angular_offset(uav: "UAV", target: "tuple[int, int]"):
    v_target = np.array(target)
    v_uav = np.array(uav.position)

    # 计算从无人机到目标的向量
    vdesired = v_target - v_uav

    # 计算无人机当前朝向和目标方向的角度
    theta_uav = math.radians(uav.direction)
    theta_target = math.atan2(vdesired[1], vdesired[0])

    # 计算从当前朝向到目标方向的角度偏移量（顺时针为正）
    theta_desired = math.degrees(theta_target - theta_uav)

    # 如果需要始终为正数（顺时针），则使用绝对值或条件判断
    # 这里保持原样，即可能为负数（表示逆时针）

    return int(theta_desired)


# 算两向量夹角
def calculate_angle_between_vectors(x, y, a, b):
    """
    计算两个向量(x, y)和(a, b)之间的夹角（以度为单位）。
    """
    # 计算点积
    dot_product = x * a + y * b

    # 计算两个向量的模长
    norm_v1 = math.sqrt(x ** 2 + y ** 2)
    norm_v2 = math.sqrt(a ** 2 + b ** 2)

    # 避免除以零的错误
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # 或者可以抛出一个异常，表示向量之一是零向量

    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = max(min(cos_theta, 1.), 0.)

    # 使用acos计算夹角（以弧度为单位），然后转换为度
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    # 计算叉积以确定方向
    cross_product = x * b - y * a
    is_counterclockwise = cross_product > 0
    if is_counterclockwise:  #
        pass
    else:
        angle_degrees = (-1) * angle_degrees
    return angle_degrees


def adjust_str_path(path, target_distance=1):
    if len(path) < 2:
        return path

    new_path = [path[0]]
    cumulative_distance = 0

    # 计算原始路径的总长度（可能已经在函数外部完成）
    total_length = sum(
        math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for ((x1, y1), (x2, y2)) in zip(path, path[1:])
    )

    # 遍历原始路径的线段
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        segment_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 计算当前线段上需要添加的点数
        num_points_in_segment = max(
            1, int(math.ceil(segment_distance / target_distance)) - 1
        )
        if num_points_in_segment > 0:
            step_length = segment_distance / (
                    num_points_in_segment + 1)  # 加上终点
            for j in range(1, num_points_in_segment + 1):
                x_new = x1 + j * step_length * (x2 - x1) / segment_distance
                y_new = y1 + j * step_length * (y2 - y1) / segment_distance
                new_path.append((x_new, y_new))
                cumulative_distance += step_length

                # 无论如何，添加线段的终点
        new_path.append((x2, y2))
        cumulative_distance += (
                segment_distance - step_length * num_points_in_segment
        )  # 避免重复计算

    return new_path


def angle_diff(angle1, angle2):
    """
        double d1 = normalize(a) - normalize(b);
    double d2 = 2 * M_PI - fabs(d1);
    if (d1 > 0) {
        d2 *= -1.0;
    }
    if (fabs(d1) < fabs(d2)) {
        return d1;
    } else {
        return d2;
    }
    """
    d1 = angle1 - angle2
    d2 = 2 * math.pi - abs(d1)
    if d1 > 0.:
        d2 *= 1.0
    if abs(d1) < abs(d2):
        return d1
    else:
        return d2


class UAV:
    max_speed = 1
    min_speed = 0.1
    max_ang_vel = 3
    length = 24
    width = 24

    v_range = (0.0, 6.0)
    w_range = (-12.0, 12.0)
    nvec = (6, 9)

    # v_range = (0.0, 3.0)
    # w_range = (-2.0, 2.0)
    # nvec = (3, 5)

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
        self.last_angular_delta = 0.
        self.env: "UAVEnvironment" = weakref.proxy(env)

        # Internal state variables
        self.x, self.y = pos
        self.dx = 0
        self.dy = 0
        self.speed = initial_speed
        self.direction = initial_direction
        self.angular_velocity = initial_angular_velocity
        self.last_linear_velocity = 0.
        self.last_angular_velocity = 0.
        self.dist_to_goal = 0.

        # Episode dependent variables
        self.goal_position = goal_position
        self.arrive = 0
        self.t = 0  # 超时计数 也是 更新临时障碍时间计数
        if self.env.use_lidar:
            self.lidar = Lidar(
                lidar_rays=self.env.lidar_rays,
                lidar_range=self.env.lidar_range,
                field_of_view=self.env.field_of_view,
            )

        self.state = None
        self.update_state()
        self.tracex = []
        self.tracey = []
        self.angular = 0
        self.angular_to_goal = 0

    @property
    def position(self):
        return self.x, self.y

    def update_state(self):  # state加主飞机前面的随机障碍
        self.dist_to_goal = math.hypot(self.x - self.goal_position[0],
                                       self.y - self.goal_position[1])
        if self.env.use_lidar:
            self.lidar_cross_pts, self.lidar_distances = self.lidar.detect(
                position=Point2d(*self.position),
                direction=self.direction,
                structures=self.env.obstacles.get_all_structures() + [Structure([
                    Point2d(0, 0),
                    Point2d(self.env.dimensions[0], 0),
                    Point2d(self.env.dimensions[0], self.env.dimensions[1]),
                    Point2d(0, self.env.dimensions[1]),
                ])] + [*(d_obstacle.get_structure() for d_obstacle in self.env.d_obstacles)]
            )
            self.lidar_distances = np.where(
                np.abs(self.lidar_distances) < 1e-8,
                0.,
                self.lidar_distances,
            )

        uav_theta_cos = math.cos(math.radians(180. - self.direction))
        uav_theta_sin = math.sin(math.radians(180. - self.direction))

        goal_dx = (self.x - self.goal_position[0]) / self.env.dimensions[0]
        goal_dy = (self.y - self.goal_position[1]) / self.env.dimensions[1]
        goal_dx_rotated = goal_dx * uav_theta_cos + goal_dy * uav_theta_sin
        goal_dy_rotated = goal_dy * uav_theta_cos - goal_dx * uav_theta_sin

        self.state = np.array([
            4 * goal_dx_rotated,
            4 * goal_dy_rotated,
            goal_dx_rotated / math.hypot(goal_dx_rotated, goal_dy_rotated),
            goal_dy_rotated / math.hypot(goal_dx_rotated, goal_dy_rotated),
            4 * self.dist_to_goal / math.hypot(*self.env.dimensions),
            self.last_linear_velocity / UAV.v_range[1],
            self.last_angular_velocity / UAV.w_range[1],
            self.last_angular_delta / UAV.w_range[1],
        ])
        if self.env.use_lidar:
            self.state = np.concatenate([self.state, self.lidar_distances])
        self.state = np.where(
            np.abs(self.state) < 1e-8,
            0.,
            self.state,
        )

        # set trace point to grid map
        # if self.env.return_raster:
        #     for tx, ty in zip(self.tracex, self.tracey):
        #         x = int(min(tx, self.env.grid_map.shape[0] - 1))
        #         y = int(min(ty, self.env.grid_map.shape[1] - 1))
        #         self.env.grid_map[x][y] = 3

    def step(self, action: "tuple[int, int]"):
        # 上一位置存储
        self.last_angular_delta = self.angular_velocity - self.last_angular_velocity
        self.last_linear_velocity = self.speed
        self.last_angular_velocity = self.angular_velocity
        last_dist_to_goal = self.dist_to_goal
        info = {"done": "not_done"}
        speed_action, turn_action = action  # 前进速度和旋转速度
        assert 0 <= speed_action < UAV.nvec[0]
        assert 0 <= turn_action < UAV.nvec[1]
        self.speed = (UAV.v_range[0]
                      + speed_action / (UAV.nvec[0]) * (UAV.v_range[1] - UAV.v_range[0]))
        if self.speed == 0.:
            self.speed = 0.1
        self.angular_velocity = (UAV.w_range[0]
                                 + turn_action / (UAV.nvec[1] - 1) * (UAV.w_range[1] - UAV.w_range[0]))
        # # Define the conversion factors
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

        self.update_state()
        info["dist_to_goal"] = self.dist_to_goal

        # reward = 0.
        crashed_dynamic = False
        crashed_fixed = False
        crashed_bounds = (
                self.x < 0
                or self.x > self.env.dimensions[0]
                or self.y < 0
                or self.y > self.env.dimensions[1]
        )  # 碰撞边界

        if self.env.obstacles.dynamic_number != 0:
            if self.dynamic_obstacle_collision() < self.length + DynamicObstacle.length:
                crashed_dynamic = True  # 碰撞动态障碍物

        if self.fixed_obstacle_collision() == 0:
            crashed_fixed = True  # 碰撞静态障碍物（如果临机障碍存在的话，也包括在内）
        # elif self.fixed_obstacle_collision() == 0.5:
        #     reward -= 10  # 若无人机警戒范围（此处大概设置为蓝色大圆弧范围）内出现静态障碍的话，扣分

        # Goal
        if self.dist_to_goal < self.length:
            self.arrive += 1
            reward_reach = 100.0
            goal_reached = True
            info["done"] = "goal_reached"
        else:
            reward_reach = 0.0
            goal_reached = False

        reward_failed = -200.0
        if crashed_bounds:
            reward = reward_failed
            info["done"] = "out_of_bounds"
        elif crashed_dynamic:
            reward = reward_failed
            info["done"] = "collision_with_dynamic_obstacle"
        elif crashed_fixed:
            reward = reward_failed
            info["done"] = "collision_with_fixed_obstacle"
        else:
            reward_const = -1.0
            reward_goal = 1.0 * (
                    last_dist_to_goal
                    - self.dist_to_goal
            ) / UAV.v_range[1]
            reward_turn_gap = -0.5 * abs(self.angular_velocity - self.last_angular_velocity) / UAV.w_range[1]
            reward_turn_direction = -0.30 * (0. if (self.angular_velocity * self.last_angular_velocity >= 0
                                                    or (self.angular_velocity == 0 and self.last_angular_velocity == 0))
                                             else 1.)
            angular_delta = self.angular_velocity - self.last_angular_velocity
            reward_turn_delta = -0.00 * (0. if (angular_delta * self.last_angular_delta >= 0)
                                         else 1.)
            reward_turn_self = 0.25 * (0.4 - abs(self.angular_velocity / UAV.w_range[1]) ** 0.5)
            reward_turn = (reward_turn_gap
                           + reward_turn_direction
                           + reward_turn_delta
                           + reward_turn_self
                           )
            reward_speed_stay = -0.10 * (0. if self.speed > 0.2 else 1.0) * (2.0 if self.angular_velocity == 0 else 1.0)
            reward_speed = reward_speed_stay
            reward = (reward_const
                      + reward_goal
                      + reward_reach
                      + reward_turn
                      + reward_speed
                      )
            # if abs(reward) > 10.:
            #     pass
        self.t += 1
        time_out = self.t >= MAX_TIMESTEP

        if self.t % 100 == 0:
            self.env.obstacles.update_occur_obstacle(self.x, self.y, self.direction)
        if self.t % 200 == 0:
            self.env.reset_dynamic_obstacles()

        done = (
                crashed_bounds
                or crashed_fixed
                or crashed_dynamic
                or goal_reached
                or time_out
        )

        if self.env.return_raster:
            obs_dict = {"observation": self.state}
            obs_dict["map"] = self.env.grid_map
        else:
            obs_dict = self.state
        return (
            obs_dict,
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
            self.x + 1. * self.width * math.cos(math.radians(self.direction + 45)),
            self.x + 1. * self.width * math.cos(math.radians(self.direction - 45)),
            self.x - 1.2 * self.width * math.cos(math.radians(self.direction + 45)),
            self.x - 1.2 * self.width * math.cos(math.radians(self.direction - 45)),
        ]
        Y = [
            self.y + 1. * self.width * -math.sin(math.radians(self.direction + 45)),
            self.y + 1. * self.width * -math.sin(math.radians(self.direction - 45)),
            self.y - 1.2 * self.width * -math.sin(math.radians(self.direction + 45)),
            self.y - 1.2 * self.width * -math.sin(math.radians(self.direction - 45)),
        ]

        obstacles = self.env.obstacles
        for i in range(obstacles.fixed_number):
            contours_list = (
                (obstacles.fixed_position[i][0],
                 obstacles.fixed_position[i][1]),
                (obstacles.fixed_position[i][0] + obstacles.fixed_wid[i],
                 obstacles.fixed_position[i][1],),
                (obstacles.fixed_position[i][0] + obstacles.fixed_wid[i],
                 obstacles.fixed_position[i][1] + obstacles.fixed_len[i],),
                (obstacles.fixed_position[i][0],
                 obstacles.fixed_position[i][1] + obstacles.fixed_len[i],),
            )
            contours_i = self.make_contours(contours_list)
            contours.append(contours_i)
        if obstacles.occur_fixed_number != 0:
            for a in range(len(obstacles.occur_fixed_position)):
                contours_list2 = (
                    (obstacles.occur_fixed_position[a][0],
                     obstacles.occur_fixed_position[a][1],),
                    (obstacles.occur_fixed_position[a][0] + obstacles.occur_fixed_wid[a],
                     obstacles.occur_fixed_position[a][1],),
                    (obstacles.occur_fixed_position[a][0] + obstacles.occur_fixed_wid[a],
                     obstacles.occur_fixed_position[a][1] + obstacles.occur_fixed_len[a],),
                    (obstacles.occur_fixed_position[a][0],
                     obstacles.occur_fixed_position[a][1] + obstacles.occur_fixed_len[a],),
                )
                contours_a = self.make_contours(contours_list2)
                contours.append(contours_a)

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
            if len(self.tracex) > 1:
                pg.draw.lines(screen,
                              (0, 0, 255),
                              False,
                              [(tx, ty) for tx, ty in zip(self.tracex, self.tracey)],
                              )
            else:
                pg.draw.circle(screen, (0, 0, 255), (self.tracex[0], self.tracey[0]), 1)
        plane_image = pg.transform.rotate(uav, self.direction)
        b_img_w, b_img_h = (plane_image.get_width(), plane_image.get_height())
        pivot = (self.x - b_img_w / 2, self.y - b_img_h / 2)
        screen.blit(plane_image, imagerect.move(pivot))
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 2)

        """机翼点,前"""
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (self.x + 1 * self.width * math.cos(math.radians(self.direction + 45)),
             self.y + 1 * self.width * -math.sin(math.radians(self.direction + 45)),),
            2,
        )
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (self.x + 1 * self.width * math.cos(math.radians(self.direction - 45)),
             self.y + 1 * self.width * -math.sin(math.radians(self.direction - 45)),),
            2,
        )

        """机翼点,后"""
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (self.x - 1.2 * self.width * math.cos(math.radians(self.direction + 45)),
             self.y - 1.2 * self.width * -math.sin(math.radians(self.direction + 45)),),
            2,
        )
        pg.draw.circle(
            screen,
            (255, 0, 0),
            (self.x - 1.2 * self.width * math.cos(math.radians(self.direction - 45)),
             self.y - 1.2 * self.width * -math.sin(math.radians(self.direction - 45)),),
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
            other_uav_dist = math.hypot(self.x - d_obstacle.x,
                                        self.y - d_obstacle.y)
            if other_uav_dist < self.length * 2:
                screen.blit(excl_mark, (self.x, self.y))

        for ii in range(obstacles.fixed_number):
            pg.draw.polygon(
                screen,
                (0, 0, 0),
                [(obstacles.fixed_position[ii][0],
                  obstacles.fixed_position[ii][1],),
                 (obstacles.fixed_position[ii][0] + obstacles.fixed_wid[ii],
                  obstacles.fixed_position[ii][1],),
                 (obstacles.fixed_position[ii][0] + obstacles.fixed_wid[ii],
                  obstacles.fixed_position[ii][1] + obstacles.fixed_len[ii],),
                 (obstacles.fixed_position[ii][0],
                  obstacles.fixed_position[ii][1] + obstacles.fixed_len[ii],),
                 (obstacles.fixed_position[ii][0],
                  obstacles.fixed_position[ii][1]), ],
            )

        for jj in range(len(obstacles.occur_fixed_position)):
            pg.draw.polygon(
                screen,
                (0, 0, 0),
                [(obstacles.occur_fixed_position[jj][0],
                  obstacles.occur_fixed_position[jj][1],),
                 (obstacles.occur_fixed_position[jj][0] + obstacles.occur_fixed_wid[jj],
                  obstacles.occur_fixed_position[jj][1],),
                 (obstacles.occur_fixed_position[jj][0] + obstacles.occur_fixed_wid[jj],
                  obstacles.occur_fixed_position[jj][1] + obstacles.occur_fixed_len[jj],),
                 (obstacles.occur_fixed_position[jj][0],
                  obstacles.occur_fixed_position[jj][1] + obstacles.occur_fixed_len[jj],),
                 (obstacles.occur_fixed_position[jj][0],
                  obstacles.occur_fixed_position[jj][1],), ],
            )
        if self.env.use_lidar and self.env.draw_lidar:
            for endpoint in self.lidar_cross_pts:
                pg.draw.lines(
                    screen,
                    (255, 255, 0),
                    False,
                    [self.position, endpoint.tuple]
                )
                pg.draw.circle(
                    screen, (255, 255, 0), endpoint.tuple, 5, 1
                )  # 画圈

    def reset(self, position, speed, direction, goal_position):
        self.x, self.y = position
        self.speed = speed
        self.direction = direction
        self.goal_position = goal_position
        self.dist_to_goal = math.hypot(self.x - self.goal_position[0],
                                       self.y - self.goal_position[1])
        self.last_linear_velocity = 0.
        self.last_angular_velocity = 0.
        self.last_angular_delta = 0.
        self.t = 0


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
        return self.x, self.y

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
            (self.x + 1 * self.length * math.cos(math.radians(self.direction)),
             self.y + 1 * self.length * -math.sin(math.radians(self.direction)),),
            2,
        )

    def get_hull(self):
        vertex_angle = math.degrees(math.atan2(DynamicObstacle.length, DynamicObstacle.width))
        vertex_angle_comp = 90. - vertex_angle
        return [
            (self.x + self.length * math.cos(math.radians(0. + self.direction + vertex_angle_comp)),
             self.y + self.length * -math.sin(math.radians(0. + self.direction + vertex_angle_comp)),),
            (self.x + self.length * math.cos(math.radians(90. + self.direction + vertex_angle)),
             self.y + self.length * -math.sin(math.radians(90. + self.direction + vertex_angle)),),
            (self.x + self.length * math.cos(math.radians(180. + self.direction + vertex_angle_comp)),
             self.y + self.length * -math.sin(math.radians(180. + self.direction + vertex_angle_comp)),),
            (self.x + self.length * math.cos(math.radians(270. + self.direction + vertex_angle)),
             self.y + self.length * -math.sin(math.radians(270. + self.direction + vertex_angle)),),
        ]

    def get_structure(self):
        vertex_angle = math.degrees(math.atan2(DynamicObstacle.length, DynamicObstacle.width))
        vertex_angle_comp = 90. - vertex_angle
        return Structure([
            Point2d(self.x + self.length * math.cos(math.radians(0. + self.direction + vertex_angle_comp)),
                    self.y + self.length * -math.sin(math.radians(0. + self.direction + vertex_angle_comp)), ),
            Point2d(self.x + self.length * math.cos(math.radians(90. + self.direction + vertex_angle)),
                    self.y + self.length * -math.sin(math.radians(90. + self.direction + vertex_angle)), ),
            Point2d(self.x + self.length * math.cos(math.radians(180. + self.direction + vertex_angle_comp)),
                    self.y + self.length * -math.sin(math.radians(180. + self.direction + vertex_angle_comp)), ),
            Point2d(self.x + self.length * math.cos(math.radians(270. + self.direction + vertex_angle)),
                    self.y + self.length * -math.sin(math.radians(270. + self.direction + vertex_angle)), ),
        ])

    def reset(self, position, speed, direction, goal_position):
        self.x, self.y = position
        self.speed = speed
        self.direction = direction
        self.goal_position = goal_position


class UAVEnvironment(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    def __init__(
            self,
            dimensions: "tuple[int, int]" = (800, 800),
            fixed_obstacles: int = 0,
            dynamic_obstacles: int = 0,
            occur_obstacles: int = 0,
            occur_number_max: int = 0,
            return_raster: bool = False,
            render_mode: str = None,
            show_windows: bool = False,
            use_lidar: bool = False,
            draw_lidar: bool = True,
            lidar_range: float = 500.,
            lidar_rays: int = 20,
            field_of_view: float = 180.,
            center_obstacles: bool = False,
            **kwargs,  # backward compatibility
    ):
        super().__init__()

        # Environmental parameters
        self.dimensions = dimensions
        self.obstacles = Obstacle(
            fixed_obstacles,
            occur_obstacles,
            occur_number_max,
            dynamic_obstacles,
        )
        self.obstacles.get_fixed_structures()

        # RL parameters
        self.return_raster = return_raster
        self.use_lidar = use_lidar
        self.draw_lidar = draw_lidar
        self.lidar_range = lidar_range
        self.lidar_rays = lidar_rays
        self.field_of_view = field_of_view
        self.center_obstacles = center_obstacles

        # Agents
        self.d_obstacles: "list[DynamicObstacle]" = []

        # raster map
        self.grid_map = None

        # Misc
        self.show_windows = show_windows
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        # Agent
        self.uav = UAV((dimensions[0] // 2, dimensions[1]), self)
        state_size = self.uav.state.size
        if self.return_raster:
            obs_space_dict = {
                "observation": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(state_size + (self.lidar_rays if self.use_lidar else 0),),
                    dtype=np.float32
                ),
                "map": gym.spaces.Box(
                    low=0,
                    high=3,
                    shape=(dimensions[1], dimensions[0]),
                    dtype=np.uint8,
                ),
            }
            self.observation_space = gym.spaces.Dict(
                obs_space_dict
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
            )
        # Use gym.spaces.Sequence() for multi waypoint
        # self.action_space = gym.spaces.MultiDiscrete([3, 5])
        self.action_space = gym.spaces.MultiDiscrete(UAV.nvec)
        self.reset()
        self.seed()

    def step(self, action: "tuple[int, int]"):
        if self.obstacles.dynamic_number != 0:
            for d_obstacle in self.d_obstacles:
                d_obstacle.step()
        obs, reward, done, info = self.uav.step(action)
        return obs, reward, done, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_dynamic_obstacles(self):
        if self.obstacles.dynamic_number != 0:
            self.d_obstacles.clear()
            self.obstacles.init_dynamic(self.uav.position, self.uav.goal_position)
            if self.obstacles.dynamic_number > 1:
                while min(*(math.hypot(
                        self.uav.x - self.obstacles.dynamic_position[idx][0],
                        self.uav.y - self.obstacles.dynamic_position[idx][1]
                ) for idx in range(self.obstacles.dynamic_number))) < self.uav.width * 20:
                    self.obstacles.init_dynamic(self.uav.position, self.uav.goal_position)
            else:
                while math.hypot(
                        self.uav.x - self.obstacles.dynamic_position[0][0],
                        self.uav.y - self.obstacles.dynamic_position[0][1]
                ) < self.uav.width * 20:
                    self.obstacles.init_dynamic(self.uav.position, self.uav.goal_position)
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

    def reset(self, *, seed=None, options=None):
        # dot_product = self.uav.goal_position[0] - uav_init_x
        # norm_v1 = math.sqrt(
        #     (self.uav.goal_position[0] - uav_init_x) ** 2
        #     + (self.uav.goal_position[1] - uav_init_y) ** 2
        # )
        # cos_theta = dot_product / norm_v1
        # angle_radians = math.acos(cos_theta)
        # angle_degrees = math.degrees(angle_radians)
        # if uav_init_y - self.uav.goal_position[1] < 0:
        #     angle_degrees = -angle_degrees
        # uav_init_angle = angle_degrees
        goal_position = (random.randint(100, 600), random.randint(100, 600))
        uav_init_x = np.random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav_init_y = np.random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav_init_v = 0
        uav_init_angle = calculate_angle_between_vectors(
            1,
            0,
            self.uav.goal_position[0] - uav_init_x,
            self.uav.goal_position[1] - uav_init_y,
        )

        self.uav.reset(
            (uav_init_x, uav_init_y),
            uav_init_v,
            uav_init_angle,
            goal_position,
        )
        while math.hypot(self.uav.x - self.uav.goal_position[0],
                         self.uav.y - self.uav.goal_position[1]) < self.dimensions[0] * 0.85:
            goal_position = (random.randint(100, 600), random.randint(100, 600))
            uav_init_x = np.random.uniform(
                self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
            )
            uav_init_y = np.random.uniform(
                self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
            )
            uav_init_v = 0
            uav_init_angle = calculate_angle_between_vectors(
                1,
                0,
                self.uav.goal_position[0] - uav_init_x,
                self.uav.goal_position[1] - uav_init_y,
            )

            self.uav.reset(
                (uav_init_x, uav_init_y),
                uav_init_v,
                uav_init_angle,
                goal_position,
            )
        self.obstacles.reset(uav_init_x, uav_init_y, uav_init_angle)

        # 静态障碍物与目标点重合或者距离过近, 重新生成
        generated_obstacle_num = 0
        self.obstacles.fixed_position.clear()
        self.obstacles.fixed_len.clear()
        self.obstacles.fixed_wid.clear()
        self.obstacles._fixed.clear()
        while generated_obstacle_num < self.obstacles.fixed_number:
            obstacle_position = (random.randint(0, 750), random.randint(0, 750))
            obstacle_len = int(np.random.uniform(10, 80))
            obstacle_wid = int(np.random.uniform(10, 80))

            o_x = obstacle_position[0] + obstacle_wid / 2
            o_y = obstacle_position[1] + obstacle_len / 2
            dist_uav = math.hypot(o_x - self.uav.x,
                                  o_y - self.uav.y)
            dist_goal = math.hypot(o_x - self.uav.goal_position[0],
                                   o_y - self.uav.goal_position[1])
            min_dist_obstacles = sum(self.dimensions)
            for i in range(generated_obstacle_num):
                o_x_1 = self.obstacles.fixed_position[i][0] + self.obstacles.fixed_wid[i] / 2
                o_y_1 = self.obstacles.fixed_position[i][1] + self.obstacles.fixed_len[i] / 2
                dist_obstacle = math.hypot(o_x_1 - o_x, o_y_1 - o_y)
                if dist_obstacle < min_dist_obstacles:
                    min_dist_obstacles = dist_obstacle
            a = self.uav.y - self.uav.goal_position[1]
            b = -self.uav.x + self.uav.goal_position[0]
            hypot_a_b = math.hypot(a, b)
            a /= hypot_a_b
            b /= hypot_a_b
            c = -a * self.uav.x - b * self.uav.y
            dist_center = abs(a * o_x + b * o_y + c)
            center_satisfied = dist_center < (
                    200 * (generated_obstacle_num + 1)) / self.obstacles.fixed_number if self.center_obstacles else True
            obstacle_satisfied = (dist_uav > UAV.length * 4
                                  and dist_goal > UAV.length * 4
                                  and min_dist_obstacles > 50
                                  and center_satisfied)
            if obstacle_satisfied:
                generated_obstacle_num += 1
                self.obstacles.fixed_position.append(obstacle_position)
                self.obstacles.fixed_len.append(obstacle_len)
                self.obstacles.fixed_wid.append(obstacle_wid)

        # 动态障碍物与目标点重合或者距离过近, 重新生成
        self.reset_dynamic_obstacles()
        if self.return_raster:
            self.grid_map = self.map_for_astar(int(uav_init_x), int(uav_init_y))
        self.uav.tracex = []
        self.uav.tracey = []

        if self.return_raster:
            obs_dict = {}
            for key in self.observation_space.spaces.keys():
                obs_dict[key] = None
            obs_dict["observation"] = self.uav.state
            obs_dict["map"] = self.grid_map
        else:
            obs_dict = self.uav.state
        return obs_dict, {}

    def map_for_astar(self, y, x):
        # Create a grid map initialized with zeros using NumPy
        grid_map = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        # Iterate over fixed obstacles and update the grid map
        for i in range(self.obstacles.fixed_number):
            x_start = max(0, self.obstacles.fixed_position[i][0] - UAV.width)
            x_end = min(
                self.dimensions[0],
                self.obstacles.fixed_position[i][0]
                + self.obstacles.fixed_wid[i]
                + UAV.width,
            )
            y_start = max(0, self.obstacles.fixed_position[i][1] - UAV.length)
            y_end = min(
                self.dimensions[0],
                self.obstacles.fixed_position[i][1]
                + self.obstacles.fixed_len[i]
                + UAV.length,
            )

            grid_map[x_start:x_end, y_start:y_end] = 1

        # Set the goal position and UAV trajectory on the grid map
        grid_map[self.uav.goal_position[0], self.uav.goal_position[1]] = 2
        grid_map[x, y] = 3

        return grid_map

    def render(self):
        if self.screen is None:
            self._init_pygame()

        self.draw()
        if self.show_windows:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
            self.clock.tick(self.metadata["render_fps"])
            pg.display.flip()

        if self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pg.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

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
            self.clock = None
            self.isopen = False

    def _init_pygame(self):
        pg.init()
        if self.show_windows:
            self.screen = pg.display.set_mode(self.dimensions)
            self.clock = pg.time.Clock()
        else:
            self.screen = pg.Surface(self.dimensions)
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


class Actor:
    def __init__(self, name):
        self.name = name
        self.mailbox = queue.Queue()
        self.alive = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def send(self, message):
        if self.alive:
            self.mailbox.put(message)

    def receive(self, message):
        # 这里可以添加处理消息的逻辑
        print(f"{self.name} received: {message}")

    def run(self):
        while self.alive:
            try:
                message = self.mailbox.get(timeout=1)  # 设置超时防止线程永远阻塞
                self.receive(message)
            except queue.Empty:
                continue

    def stop(self):
        self.alive = False
        self.thread.join()


def astar_test(env: UAVEnvironment, episodes: int = 3, render: bool = False):
    actor = Actor("Astar path")
    for i in range(episodes):
        env.reset()
        done = False
        # plt.cla()
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
            # 发送消息
            actor.send(path)
            re_path = [(y, x) for x, y in path]
            print(re_path)

        for p in path:
            env.grid_map[int(p[0]) - 1][int(p[1]) - 1] = 3

        follow_position = 0
        adjusted_path = adjust_str_path(re_path, 1)
        pre_point = adjusted_path[0]
        # print('new_path', adjusted_path)
        # for p in adjusted_path:
        #     env.grid_map[int(p[1]) - 1][int(p[0]) - 1] = 4
        # chazhi = math.hypot(p[0] - pre_point[0],  # 插值之后并不一定距离为1
        #                     p[1] - pre_point[1])
        # pre_point = p
        # print(chazhi)
        print("AStar runtime:", end_time - start_time)
        # plt.imshow(env.grid_map, cmap="coolwarm", interpolation="nearest")
        # plt.colorbar()
        # plt.savefig('Astar{}'.format(i))
        # plt.show()

        action_list = []
        delta_x = adjusted_path[1][0] - adjusted_path[0][0]
        delta_y = adjusted_path[1][1] - adjusted_path[0][1]
        delta_goalx = env.uav.goal_position[0] - env.uav.x
        delta_goaly = env.uav.goal_position[1] - env.uav.y
        theta0 = calculate_angle_between_vectors(
            delta_goalx,
            delta_goaly,
            delta_x,
            delta_y)
        print('初始角度', theta0)
        if theta0 != 0:
            # if delta_goalx > 0:
            #     print('>0')
            #     if delta_goaly > 0:
            #         print('goaly>0')
            #         theta0 = round(theta0) - 90
            #     else:
            #         print('goaly<0')
            #         theta0 = round(theta0) - 45
            # else:
            #     print('<0')
            #     if delta_goaly > 0:
            #         print('goaly>0')
            #         theta0 = round(theta0) + 45
            #     else:
            #         print('goaly<0')
            #         theta0 = round(theta0)
            angular_action = theta0 / abs(theta0)
            for t in range(0, round(abs(theta0))):
                action_list.append((0, int(-angular_action) + 2))
            # print('初始旋转',action_list)
        for fp in range(len(adjusted_path)):
            if fp < len(adjusted_path) - 2:
                delta_x1 = adjusted_path[fp + 1][0] - adjusted_path[fp][0]
                delta_y1 = adjusted_path[fp + 1][1] - adjusted_path[fp][1]
                delta_x2 = adjusted_path[fp + 2][0] - adjusted_path[fp + 1][0]
                delta_y2 = adjusted_path[fp + 2][1] - adjusted_path[fp + 1][1]
                theta = calculate_angle_between_vectors(
                    delta_x1,
                    delta_y1,
                    delta_x2,
                    delta_y2, )
                action_list.append((1, 2))
                if round(theta) == 0:  # 拐点判定
                    pass
                else:
                    theta = round(theta)
                    # print('拐弯', theta)
                    angular_action = theta / abs(theta)
                    for t in range(0, round(abs(theta))):
                        action_list.append((0, int(-angular_action) + 2))
                    # action_list.append()

                fp += 1
            else:
                action_list.append((1, 2))
                # print('new path', adjusted_path)
        # 发送消息
        actor.send(action_list)
        while not done:
            # plt.cla()
            action = action_list[follow_position]
            follow_position += 1
            _, _, done, _, info = env.step(action)
            if render:
                env.render()
            else:
                print(f"Episode {i} done due to {info['done']}")
        env.close()


def rl_test(env: UAVEnvironment, episodes: int = 3, render: bool = False):
    for i in range(episodes):
        env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            # action = [1, 4]
            obs, reward, done, _, info = env.step(action)
            # print(obs)
            print(reward)
            if render:
                env.render()
        else:
            print(f"Episode {i} done due to {info['done']}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--astar", action="store_true")
    parser.add_argument("--width", type=int, default=800)  # 窗口大小
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--fixed_obstacle_number", type=int, default=7)  # 静态障碍数量
    parser.add_argument("--occur_fixed_number", type=int, default=1)  # 临时机制障碍数量
    parser.add_argument(
        "--occur_number_max", type=int, default=3
    )  # 临时机制障碍最大同时存在数量
    parser.add_argument(
        "--dynamic_obstacle_number", type=int, default=0
    )  # 动态障碍数量
    args = parser.parse_args()

    dim = (args.width, args.height)
    env = UAVEnvironment(
        dim,
        args.fixed_obstacle_number,
        args.dynamic_obstacle_number,
        args.occur_fixed_number,
        args.occur_number_max,
        return_raster=True,
        show_windows=True,
        use_lidar=True,
        draw_lidar=False,
        lidar_range=250,
        lidar_rays=21,
        field_of_view=210,
        center_obstacles=True,
        # render_mode='rgb_array',
    )
    # env = PretextWrapper(env, pretext_dir=None, device='cpu')
    # env = RasterWrapper(env)
    # env = ActionSpaceWrapper(env)

    # args.astar = True
    if args.astar:
        astar_test(env, args.episodes, args.render)
    else:
        rl_test(env, args.episodes, args.render)
