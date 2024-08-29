from __future__ import annotations

import math
import os
import time
import weakref

import cv2
import gymnasium as gym
import numpy as np
import pygame as pg
import pygame.freetype as pgft
from gymnasium.utils import seeding

from uav_envs.utils.lidar import Point2d, Structure, Lidar

# from scipy.interpolate import splev, splrep
# from scipy.optimize import curve_fit

root = os.path.dirname(os.path.abspath(__file__))
uav = pg.image.load(os.path.join(root, "res/uav2.jpg"))
uav = pg.transform.rotozoom(uav, 0, 0.035)

background = pg.image.load(os.path.join(root, "res/sky.png"))
background = pg.transform.rotozoom(background, 0, 2.2)
imagerect = uav.get_rect()

missile = pg.image.load(os.path.join(root, "res/balloon.png"))
missile = pg.transform.rotozoom(missile, 0, 0.0014)
imagerect_missile = missile.get_rect()

pgft.init()
font = pgft.SysFont("", 20)
excl_mark, _ = font.render("!!!", (255, 0, 0))

STATE_SIZE = 5
MAX_TIMESTEP = 300
OCCUR_LEN_RATIO = 10

FIXED_OBSTACLES_SIZES = (30, 60)
OCCUR_OBSTACLES_SIZES = (20, 40)

DYNAMIC_SCALE_AVOID = 3.0
DYNAMIC_SCALE_DETECT = 10.0
DYNAMIC_SPEED = (3, 4)
DANGER_ZONE_SCALE = 2.5


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

    def reset(self, x, y, dire, seed=None):
        self.np_random, seed = seeding.np_random(seed)
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
                    (self.np_random.integers(0, 750), self.np_random.integers(0, 750))
                )
                self.fixed_len.append(int(self.np_random.uniform(*FIXED_OBSTACLES_SIZES)))
                self.fixed_wid.append(int(self.np_random.uniform(*FIXED_OBSTACLES_SIZES)))
        pass

    def init_occur_fixed(self, x, y, dire):
        self.occur_fixed_position.clear()
        self.occur_fixed_angular.clear()
        self.occur_fixed_len.clear()
        self.occur_fixed_wid.clear()
        self._occur.clear()

        if self.occur_fixed_number != 0:
            # 固定障碍物
            for i in range(self.occur_fixed_number):
                occur_len = int(self.np_random.uniform(*OCCUR_OBSTACLES_SIZES))
                self.occur_fixed_len.append(occur_len)
                occur_wid = int(self.np_random.uniform(*OCCUR_OBSTACLES_SIZES))
                self.occur_fixed_wid.append(occur_wid)
                self.occur_fixed_angular.append(
                    int(self.np_random.uniform(dire - 90, dire + 90))
                )
                self.occur_fixed_position.append(
                    (
                        round(
                            x
                            + OCCUR_LEN_RATIO
                            * UavAgent.length
                            * math.cos(math.radians(self.occur_fixed_angular[-1]))
                            - occur_wid / 2
                        ),
                        round(
                            y
                            + OCCUR_LEN_RATIO
                            * UavAgent.length
                            * -math.sin(math.radians(self.occur_fixed_angular[-1]))
                            - occur_len / 2
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
            occur_len = int(self.np_random.uniform(*OCCUR_OBSTACLES_SIZES))
            self.occur_fixed_len.append(occur_len)
            occur_wid = int(self.np_random.uniform(*OCCUR_OBSTACLES_SIZES))
            self.occur_fixed_wid.append(occur_wid)
            self.occur_fixed_angular.append(
                int(self.np_random.uniform(dire - 90, dire + 90))
            )
            self.occur_fixed_position.append(
                (
                    round(
                        x
                        + OCCUR_LEN_RATIO
                        * UavAgent.length
                        * math.cos(math.radians(self.occur_fixed_angular[-1]))
                        - occur_wid / 2
                    ),
                    round(
                        y
                        + OCCUR_LEN_RATIO
                        * UavAgent.length
                        * -math.sin(math.radians(self.occur_fixed_angular[-1]))
                        - occur_len / 2
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
                dynamic_x = self.np_random.integers(50, 750)
                dynamic_y = self.np_random.integers(50, 750)
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
                if round(min_angular) == round(max_angular):
                    max_angular += 1
                self.dynamic_direction.append(
                    self.np_random.integers(round(min_angular),
                                            round(max_angular)))
                self.dynamic_speed.append(self.np_random.integers(*DYNAMIC_SPEED) * 0.5)
                self.dynamic_angular_velocity.append(self.np_random.integers(0, 100) * 0.01)
                # self.dynamic_mode.append(self.np_random.integers(0, 1))  # 0直线行驶, 1绕圈
                self.dynamic_mode.append(2)  # 0直线行驶, 1绕圈. 2随机

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


def angular_offset(uav: "UavAgent", target: "tuple[int, int]"):
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
    cos_theta = max(min(cos_theta, 1.), -1.)

    # 使用acos计算夹角（以弧度为单位），然后转换为度
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)
    # 计算叉积以确定方向
    cross_product = x * b - y * a
    is_counterclockwise = cross_product > 0
    if not is_counterclockwise:
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


class UavAgent:
    max_speed = 1
    min_speed = 0.1
    max_ang_vel = 3
    length = 6
    width = 6

    v_range = (0.0, 6.0)
    w_range = (-12.0, 12.0)
    nvec = (6, 9)

    # v_range = (0.0, 2.0)
    # w_range = (-2.0, 2.0)
    # nvec = (3, 5)

    def __init__(
            self,
            pos: "tuple[int, int]",
            env: "UavEnvironment",
            initial_speed: float = 0,
            initial_direction: float = 0,
            initial_angular_velocity: float = 0,
            goal_position: "tuple[int, int]" = (
                    300,
                    300,
            ),
    ):
        self.last_angular_delta = 0.
        self.env: "UavEnvironment" = weakref.proxy(env)

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
            self.last_linear_velocity / UavAgent.v_range[1],
            self.last_angular_velocity / UavAgent.w_range[1],
            self.last_angular_delta / UavAgent.w_range[1],
        ])
        if self.env.use_lidar:
            self.state = np.concatenate([self.state, self.lidar_distances])
        self.state = np.where(
            np.abs(self.state) < 1e-8,
            0.,
            self.state,
        )

    def step(self, action: "tuple[int, int]"):
        # 上一位置存储
        self.last_angular_delta = self.angular_velocity - self.last_angular_velocity
        self.last_linear_velocity = self.speed
        self.last_angular_velocity = self.angular_velocity
        last_dist_to_goal = self.dist_to_goal
        info = {"done": "not_done"}
        speed_action, turn_action = action  # 前进速度和旋转速度
        assert 0 <= speed_action < UavAgent.nvec[0]
        assert 0 <= turn_action < UavAgent.nvec[1]
        if self.env.prevent_stiff:
            self.speed = (UavAgent.v_range[0]
                          + (speed_action + 1) / (UavAgent.nvec[0]) * (UavAgent.v_range[1] - UavAgent.v_range[0]))
        else:
            self.speed = (UavAgent.v_range[0]
                          + speed_action / (UavAgent.nvec[0] - 1) * (UavAgent.v_range[1] - UavAgent.v_range[0]))
            if self.speed == 0.:
                self.speed = 0.1
        self.angular_velocity = (UavAgent.w_range[0]
                                 + turn_action / (UavAgent.nvec[1] - 1) * (UavAgent.w_range[1] - UavAgent.w_range[0]))

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
        crashed_bounds = any((
                                     x_ < 0
                                     or x_ > self.env.dimensions[0]
                                     or y_ < 0
                                     or y_ > self.env.dimensions[1]
                             ) for x_, y_ in zip(X, Y))  # 碰撞边界

        reward_danger_zone = 0.
        if self.env.obstacles.dynamic_number != 0:
            dist_to_dynamic = self.dynamic_obstacle_collision()
            if dist_to_dynamic < self.length + DynamicObstacle.length:
                crashed_dynamic = True  # 碰撞动态障碍物
            elif dist_to_dynamic < self.length + DANGER_ZONE_SCALE * DynamicObstacle.length:
                reward_danger_zone = 1.

        if self.fixed_obstacle_collision() == 0:
            crashed_fixed = True  # 碰撞静态障碍物（如果临机障碍存在的话，也包括在内）
        # elif self.fixed_obstacle_collision() == 0.5:
        #     reward -= 10  # 若无人机警戒范围（此处大概设置为蓝色大圆弧范围）内出现静态障碍的话，扣分

        # Goal
        if self.dist_to_goal < 3 * self.length:
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
            ) / UavAgent.v_range[1]
            reward_turn_gap = -0.5 * abs(self.angular_velocity - self.last_angular_velocity) / UavAgent.w_range[1]
            reward_turn_direction = -0.30 * (0. if (self.angular_velocity * self.last_angular_velocity >= 0
                                                    or (self.angular_velocity == 0 and self.last_angular_velocity == 0))
                                             else 1.)
            angular_delta = self.angular_velocity - self.last_angular_velocity
            reward_turn_delta = -0.00 * (0. if (angular_delta * self.last_angular_delta >= 0)
                                         else 1.)
            reward_turn_self = 0.25 * (0.4 - abs(self.angular_velocity / UavAgent.w_range[1]) ** 0.5)
            reward_turn = (reward_turn_gap
                           + reward_turn_direction
                           + reward_turn_delta
                           + reward_turn_self
                           )
            if not self.env.prevent_stiff:
                reward_speed_stay = (-0.10
                                     * (0. if self.speed > 0.2 else 1.0)
                                     * (2.0 if self.angular_velocity == 0 else 1.0))
            else:
                reward_speed_stay = 0.
            reward_speed = reward_speed_stay
            reward = (reward_const
                      + reward_danger_zone
                      + reward_goal
                      + reward_reach
                      + reward_turn
                      + reward_speed
                      )
            # if abs(reward) > 10.:
            #     pass
        self.t += 1
        time_out = self.t >= MAX_TIMESTEP
        if time_out:
            info["done"] = "out_of_time"

        if self.t % 100 == 0:
            self.env.obstacles.update_occur_obstacle(self.x, self.y, self.direction)
        # if self.t % 200 == 0:
        #     self.env.reset_dynamic_obstacles()

        done = (
                crashed_bounds
                or crashed_fixed
                or crashed_dynamic
                or goal_reached
                or time_out
        )

        obs = self.state
        return (
            obs,
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
                    if max(dist2) <= -self.length * OCCUR_LEN_RATIO:
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
            screen, (255, 51, 255), (int(self.x), int(self.y)), self.length * OCCUR_LEN_RATIO, 1
        )  # 画圈，动态障碍出现圈，也是警戒范围？
        pg.draw.circle(
            screen,
            (0, 255, 0),
            (int(self.goal_position[0]), int(self.goal_position[1])),
            3 * self.length,
            2,
        )  # 目标绿圈

        obstacles = self.env.obstacles
        for d_obstacle in self.env.d_obstacles:
            other_uav_dist = math.hypot(self.x - d_obstacle.x,
                                        self.y - d_obstacle.y)
            if other_uav_dist < self.length + DANGER_ZONE_SCALE * DynamicObstacle.length:
                screen.blit(excl_mark, (self.x, self.y))

        for ii in range(obstacles.fixed_number):
            pg.draw.polygon(
                screen,
                (105, 105, 105),
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
                (105, 105, 105),
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
    length = 5
    width = 5

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

    @property
    def position(self):
        return self.x, self.y

    def step(self):
        match self.mode:
            case 0:
                self.dx = self.speed * math.cos(math.radians(self.direction))
                self.dy = self.speed * -math.sin(math.radians(self.direction))
                self.x += self.dx
                self.y += self.dy
                self.direction = (self.direction + self.angular_velocity) % 360
            case 1:
                self.speed = self.speed
                self.angular_velocity = 0
                self.dx = self.speed * math.cos(math.radians(self.direction))
                self.dy = self.speed * -math.sin(math.radians(self.direction))
                self.x += self.dx
                self.y += self.dy
                self.direction = (self.direction + self.angular_velocity) % 360
            case 2:
                angular_velocity = self.np_random.integers(-60, 60)
                self.direction = (self.direction + angular_velocity) % 360
                self.dx = self.speed * math.cos(math.radians(self.direction))
                self.dy = self.speed * -math.sin(math.radians(self.direction))
                self.x += self.dx
                self.y += self.dy
            case _:
                raise ValueError(f"There is no such mode {self.mode} for dynamic obstalces")

    def draw(self, screen: pg.Surface):
        pg.draw.circle(screen, (255, 137, 138), (int(self.x), int(self.y)), self.width * DANGER_ZONE_SCALE, width=0)
        pg.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.width, width=0)

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

    def reset(self, position, speed, direction, goal_position, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.x, self.y = position
        self.speed = speed
        self.direction = direction
        self.goal_position = goal_position


class UavEnvironment(gym.Env):
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
            prevent_stiff: bool = True,
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
        self.prevent_stiff = prevent_stiff
        self.use_lidar = use_lidar
        self.draw_lidar = draw_lidar
        self.lidar_range = lidar_range
        self.lidar_rays = lidar_rays
        self.field_of_view = field_of_view
        self.center_obstacles = center_obstacles
        self.visited = None
        self.dynamic_avoids = []

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
        self.uav = UavAgent((dimensions[0] // 2, dimensions[1]), self)
        state_size = self.uav.state.size
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
            )
        })
        self.action_space = gym.spaces.Discrete(UavAgent.nvec[0] * UavAgent.nvec[1])

    def step(self, action: int):
        if self.obstacles.dynamic_number != 0:
            for d_obstacle in self.d_obstacles:
                d_obstacle.step()
        speed_action, turn_action = action // UavAgent.nvec[-1], action % UavAgent.nvec[-1]
        obs, reward, done, info = self.uav.step((speed_action, turn_action))
        obs = {
            'observation': obs,
            'succeeded': info['done'] == 'goal_reached',
        }
        for key in self.observation_space.spaces.keys():
            if key not in obs:
                obs[key] = None
        return obs, reward, done, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_dynamic_obstacles(self, seed=None):
        if self.obstacles.dynamic_number != 0:
            self.d_obstacles.clear()
            self.obstacles.init_dynamic(self.uav.position, self.uav.goal_position)
            if self.obstacles.dynamic_number > 1:
                while min(*(math.hypot(
                        self.uav.x - self.obstacles.dynamic_position[idx][0],
                        self.uav.y - self.obstacles.dynamic_position[idx][1]
                ) for idx in range(self.obstacles.dynamic_number))) < self.uav.width * 10 or min(*(math.hypot(
                    self.uav.goal_position[0] - self.obstacles.dynamic_position[idx][0],
                    self.uav.goal_position[1] - self.obstacles.dynamic_position[idx][1]
                ) for idx in range(self.obstacles.dynamic_number))) < self.uav.width * 10:
                    self.obstacles.init_dynamic(self.uav.position, self.uav.goal_position)
            else:
                while math.hypot(
                        self.uav.x - self.obstacles.dynamic_position[0][0],
                        self.uav.y - self.obstacles.dynamic_position[0][1]
                ) < self.uav.width * 10 or math.hypot(
                    self.uav.goal_position[0] - self.obstacles.dynamic_position[0][0],
                    self.uav.goal_position[1] - self.obstacles.dynamic_position[0][1]
                ) < self.uav.width * 10:
                    self.obstacles.init_dynamic(self.uav.position, self.uav.goal_position)
            for i in range(self.obstacles.dynamic_number):
                init_x = self.obstacles.dynamic_position[i][0]
                init_y = self.obstacles.dynamic_position[i][1]
                init_v = self.obstacles.dynamic_speed[i]
                init_angle = self.obstacles.dynamic_direction[i]
                d_obstacle = DynamicObstacle(self.obstacles.dynamic_obstacle[i], mode=self.obstacles.dynamic_mode[i])

                d_obstacle.reset(
                    (init_x, init_y),
                    init_v,
                    init_angle,
                    (0, 0),
                    seed=seed,
                )
                self.d_obstacles.append(d_obstacle)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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
        self.visited = None
        self.dynamic_avoids = []
        goal_position = (self.np_random.integers(100, self.dimensions[0] - 100),
                         self.np_random.integers(100, self.dimensions[1] - 100))
        uav_init_x = self.np_random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav_init_y = self.np_random.uniform(
            self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
        )
        uav_init_v = 0
        uav_init_angle = -calculate_angle_between_vectors(
            1,
            0,
            goal_position[0] - uav_init_x,
            goal_position[1] - uav_init_y,
        )
        # uav_init_angle = -0
        self.uav.reset(
            (uav_init_x, uav_init_y),
            uav_init_v,
            uav_init_angle,
            goal_position,
        )
        while math.hypot(self.uav.x - self.uav.goal_position[0],
                         self.uav.y - self.uav.goal_position[1]) < min(self.dimensions) * 0.75:
            goal_position = (self.np_random.integers(100, 600), self.np_random.integers(100, 600))
            uav_init_x = self.np_random.uniform(
                self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
            )
            uav_init_y = self.np_random.uniform(
                self.dimensions[0] * 0.1, self.dimensions[0] * 0.9
            )
            uav_init_v = 0
            # uav_init_angle = 0
            uav_init_angle = -calculate_angle_between_vectors(
                1,
                0,
                goal_position[0] - uav_init_x,
                goal_position[1] - uav_init_y,
            )

            self.uav.reset(
                (uav_init_x, uav_init_y),
                uav_init_v,
                uav_init_angle,
                goal_position,
            )
        self.obstacles.reset(uav_init_x, uav_init_y, uav_init_angle, seed=seed)

        # 静态障碍物与目标点重合或者距离过近, 重新生成
        generated_obstacle_num = 0
        self.obstacles.fixed_position.clear()
        self.obstacles.fixed_len.clear()
        self.obstacles.fixed_wid.clear()
        self.obstacles._fixed.clear()
        if isinstance(options, dict) and 'center' in options:
            center_condition = options['center']
        else:
            center_condition = self.center_obstacles
        while generated_obstacle_num < self.obstacles.fixed_number:
            obstacle_position = (self.np_random.integers(0, self.dimensions[0] - 50),
                                 self.np_random.integers(0, self.dimensions[1] - 100))
            obstacle_len = int(self.np_random.uniform(*FIXED_OBSTACLES_SIZES))
            obstacle_wid = int(self.np_random.uniform(*FIXED_OBSTACLES_SIZES))

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
                    200 * (generated_obstacle_num + 1)) / self.obstacles.fixed_number if center_condition else True
            obstacle_satisfied = (dist_uav > UavAgent.length * OCCUR_LEN_RATIO
                                  and dist_goal > UavAgent.length * OCCUR_LEN_RATIO
                                  and min_dist_obstacles > 50
                                  and center_satisfied)
            if obstacle_satisfied:
                generated_obstacle_num += 1
                self.obstacles.fixed_position.append(obstacle_position)
                self.obstacles.fixed_len.append(obstacle_len)
                self.obstacles.fixed_wid.append(obstacle_wid)

        # 动态障碍物与目标点重合或者距离过近, 重新生成
        self.reset_dynamic_obstacles(seed=seed)
        self.uav.tracex = []
        self.uav.tracey = []

        obs = self.uav.state
        obs = {
            'observation': obs,
            'succeeded': False,
        }
        for key in self.observation_space.spaces.keys():
            if key not in obs:
                obs[key] = None
        return obs, {}

    def map_for_astar(self):
        uav_scale = 2.5
        if self.visited is None:
            self.visited = [False] * len(self.d_obstacles)
        # Create a grid map initialized with zeros using NumPy
        grid_map = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        # Iterate over fixed obstacles and update the grid map
        for i in range(self.obstacles.fixed_number):
            x_start = max(0, self.obstacles.fixed_position[i][0] - int(uav_scale * UavAgent.width))
            x_end = min(
                self.dimensions[0],
                self.obstacles.fixed_position[i][0]
                + self.obstacles.fixed_wid[i]
                + int(uav_scale * UavAgent.width),
            )
            y_start = max(0, self.obstacles.fixed_position[i][1] - int(uav_scale * UavAgent.width))
            y_end = min(
                self.dimensions[1],
                self.obstacles.fixed_position[i][1]
                + self.obstacles.fixed_len[i]
                + int(uav_scale * UavAgent.width),
            )
            grid_map[y_start:y_end, x_start:x_end] = 1
        for i in range(len(self.obstacles.occur_fixed_position)):
            x_start = max(0, self.obstacles.occur_fixed_position[i][0] - int(uav_scale * UavAgent.width))
            x_end = min(
                self.dimensions[0],
                self.obstacles.occur_fixed_position[i][0]
                + self.obstacles.occur_fixed_wid[i]
                + int(uav_scale * UavAgent.width),
            )
            y_start = max(0, self.obstacles.occur_fixed_position[i][1] - int(uav_scale * UavAgent.width))
            y_end = min(
                self.dimensions[1],
                self.obstacles.occur_fixed_position[i][1]
                + self.obstacles.occur_fixed_len[i]
                + int(uav_scale * UavAgent.width),
            )
            grid_map[y_start:y_end, x_start:x_end] = 1
        for idx, d_obstacle in enumerate(self.d_obstacles):
            d_dist = math.hypot(self.uav.x - d_obstacle.x, self.uav.y - d_obstacle.y)
            if d_dist < DYNAMIC_SCALE_DETECT * DynamicObstacle.length and not self.visited[idx]:
                self.visited[idx] = True
                x_start = max(0,
                              int(d_obstacle.x - DYNAMIC_SCALE_AVOID * DynamicObstacle.length - uav_scale * UavAgent.width))
                x_end = min(
                    self.dimensions[0],
                    int(d_obstacle.x + DYNAMIC_SCALE_AVOID * DynamicObstacle.length + uav_scale * UavAgent.width),
                )
                y_start = max(0,
                              int(d_obstacle.y - DYNAMIC_SCALE_AVOID * DynamicObstacle.length - uav_scale * UavAgent.width))
                y_end = min(
                    self.dimensions[1],
                    int(d_obstacle.y + DYNAMIC_SCALE_AVOID * DynamicObstacle.length + uav_scale * UavAgent.width),
                )
                self.dynamic_avoids.append((y_start, y_end, x_start, x_end))
        for y_start, y_end, x_start, x_end in self.dynamic_avoids:
            grid_map[y_start:y_end, x_start:x_end] = 1
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
            (255, 255, 255),
            pg.Rect(0, 0, self.dimensions[0], self.dimensions[1]),
        )
        # self.screen.blit(background, (0, 0))
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
        self.screen.blit(font.render(f'Step: {self.uav.t:03d}', (0, 0, 0))[0], (16, 40))

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


if __name__ == "__main__":
    def rl_test(env: UavEnvironment, episodes: int = 3, render: bool = False):
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


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--width", type=int, default=1000)  # 窗口大小
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--fixed_obstacle_number", type=int, default=20)  # 静态障碍数量
    parser.add_argument("--occur_fixed_number", type=int, default=1)  # 临时机制障碍数量
    parser.add_argument(
        "--occur_number_max", type=int, default=3
    )  # 临时机制障碍最大同时存在数量
    parser.add_argument(
        "--dynamic_obstacle_number", type=int, default=20
    )  # 动态障碍数量
    args = parser.parse_args()

    dim = (args.width, args.height)
    env = UavEnvironment(
        dim,
        args.fixed_obstacle_number,
        args.dynamic_obstacle_number,
        args.occur_fixed_number,
        args.occur_number_max,
        prevent_stiff=False,
        show_windows=True,
        use_lidar=True,
        draw_lidar=False,
        lidar_range=250,
        lidar_rays=21,
        field_of_view=210,
        center_obstacles=False,
    )
    rl_test(env, args.episodes, args.render)
