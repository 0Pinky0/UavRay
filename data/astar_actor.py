from __future__ import annotations

import math

from envs.uav_env_v7 import UavAgent, UavEnvironment, calculate_angle_between_vectors, DynamicObstacle, \
    DYNAMIC_SCALE_DETECT


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
    def __init__(self, grid, start, goal, step_limit=1000):
        self.grid = grid
        self.start = Node(None, start)
        self.goal = goal
        self.open_set: "list[Node]" = []
        self.closed_set = set()
        self.other_b = 0  # A*未加入动态障碍信息
        self.step_limit = step_limit

    def heuristic(self, a, b):
        # 使用曼哈顿距离作为启发式函数
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)
        # return math.hypot((x1 - x2), (y1 - y2))

    def get_neighbors(self, node: Node):
        neighbors = []
        directions = []
        row, col = node.position
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    # speed = (UavAgent.v_range[0]
                    #          + 1 / (UavAgent.nvec[0] - 1) * (UavAgent.v_range[1] - UavAgent.v_range[0]))
                    speed = 6
                    directions.append((speed * i / math.hypot(i, j),
                                       speed * j / math.hypot(i, j)))

        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            # 增加动态障碍other_b
            mapp = self.grid

            # 检查新坐标是否在网格范围内
            if 0 <= new_row < len(mapp) and 0 <= new_col < len(mapp[0]):
                # 检查新坐标是否不是障碍物
                new_row_ = int(new_row)
                new_col_ = int(new_col)
                if (
                        mapp[new_row_][new_col_] != 1
                        and (new_row_, new_col_) != self.start.position
                ):
                    new_node = Node(node, (new_row, new_col))
                    neighbors.append(new_node)

        return neighbors

    def a_star_search(self):
        self.open_set.append(self.start)
        steps = self.step_limit
        while self.open_set and steps > 0:
            steps -= 1
            # 选择f值最小的节点
            current = min(self.open_set, key=lambda node: node.f)
            self.open_set.remove(current)
            self.closed_set.add(current)

            if math.hypot(current.position[0] - self.goal[0],
                          current.position[1] - self.goal[1]) < 0.7 * UavAgent.length:
                # print(f'step: {self.step_limit - steps:4d}', end=' ')
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


def astar_solver(env: UavEnvironment, step_limit=1000):
    astar = AStar(
        env.map_for_astar(),
        (env.uav.y, env.uav.x),
        (env.uav.goal_position[1], env.uav.goal_position[0]),
        step_limit,
    )
    path = astar.a_star_search()

    if path is None:
        return None
    else:
        re_path = [(y, x) for x, y in path]

    adjusted_path = re_path

    action_list = []
    delta_x = adjusted_path[1][0] - adjusted_path[0][0]
    delta_y = adjusted_path[1][1] - adjusted_path[0][1]
    # delta_goalx = env.uav.goal_position[0] - env.uav.x
    # delta_goaly = env.uav.goal_position[1] - env.uav.y
    delta_goalx = math.cos(math.radians(env.uav.direction))
    delta_goaly = -math.sin(math.radians(env.uav.direction))
    # delta_goalx = -math.sin(math.radians(env.uav.direction))
    # delta_goaly = math.cos(math.radians(env.uav.direction))
    theta0 = calculate_angle_between_vectors(
        delta_goalx,
        delta_goaly,
        delta_x,
        delta_y)
    turn_gap = round((UavAgent.w_range[1] - UavAgent.w_range[0]) / (UavAgent.nvec[1] - 1))
    max_turn = (UavAgent.nvec[1] - 1) // 2
    if theta0 != 0:
        angular_action = theta0 / abs(theta0)
        num_turns = math.floor(abs(theta0) // turn_gap)
        if abs(num_turns * turn_gap - abs(theta0)) > abs((num_turns + 1) * turn_gap - abs(theta0)):
            num_turns += 1
        while num_turns > 0:
            for j in range(max_turn, 0, -1):
                if j <= num_turns:
                    num_turns -= j
                    action_list.append((0, int(-angular_action * j) + (UavAgent.nvec[1] - 1) // 2))
                    break
        #     action_list.extend([(0, int(-angular_action) + (UavAgent.nvec[1] - 1) // 2)] * num_turns)
        # action_list.extend([(0, int(-angular_action) + (UavAgent.nvec[1] - 1) // 2)] * num_turns)
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
            action_list.append((5, (UavAgent.nvec[1] - 1) // 2))
            if round(theta) == 0:  # 拐点判定
                pass
            else:
                theta = round(theta)
                angular_action = theta / abs(theta)
                num_turns = math.floor(abs(theta) // turn_gap)
                if abs(num_turns * turn_gap - abs(theta)) > abs((num_turns + 1) * turn_gap - abs(theta)):
                    num_turns += 1
                while num_turns > 0:
                    for j in range(max_turn, 0, -1):
                        if j <= num_turns:
                            num_turns -= j
                            action_list.append((0, int(-angular_action * j) + (UavAgent.nvec[1] - 1) // 2))
                            break
                # action_list.extend([(0, int(-angular_action) + (UavAgent.nvec[1] - 1) // 2)] * num_turns)
            fp += 1
        else:
            action_list.append((5, (UavAgent.nvec[1] - 1) // 2))
    action_list = [*(action[0] * UavAgent.nvec[1] + action[1] for action in action_list)]
    return action_list


class AstarActor:
    straight_action = 5 * UavAgent.nvec[1] + (UavAgent.nvec[1] - 1) // 2

    def __init__(self, env, step_limit: int = 1000, allow_shutdown: bool = False):
        self.env = env
        self.step_limit = step_limit
        self.action_list = []
        self.action_index = 0
        self.visited = None
        self.iter = 0
        self.need_reschedule = True
        self.reschedule_time = 0
        self.allow_shutdown = allow_shutdown

    def get_action(self):
        if self.need_reschedule:
            attempted_action_list = astar_solver(self.env, self.step_limit)
            if attempted_action_list is not None:
                self.action_list = attempted_action_list
                self.action_index = 0
            else:
                self.action_list = []
                if self.allow_shutdown:
                    self.action_list.append(-1)
                self.action_index = 0
            self.need_reschedule = False
            self.reschedule_time += 1
        if self.action_index < len(self.action_list):
            next_action = self.action_list[self.action_index]
            self.action_index += 1
        else:
            next_action = self.straight_action
        self.iter += 1
        if self.iter >= 100:
            self.need_reschedule = True
            self.iter = 0
        if self.visited is None:
            self.visited = [False] * len(self.env.d_obstacles)
        for idx, d_obstacle in enumerate(self.env.d_obstacles):
            d_dist = math.hypot(self.env.uav.x - d_obstacle.x, self.env.uav.y - d_obstacle.y)
            if d_dist < DYNAMIC_SCALE_DETECT * DynamicObstacle.length and not self.visited[idx]:
                self.need_reschedule = True
                self.visited[idx] = True
        return next_action

    def reset(self):
        self.action_list = []
        self.action_index = 0
        self.visited = None
        self.iter = 0
        self.need_reschedule = True
        self.reschedule_time = 0
