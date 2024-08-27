import math
import numpy as np


class Point2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other: "Point2d") -> "Point2d":
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2d") -> "Point2d":
        return Point2d(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int | float) -> "Point2d":
        return Point2d(self.x * other, self.y * other)

    def __mod__(self, other: "Point2d") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    @property
    def len(self):
        return math.hypot(self.x, self.y)

    @property
    def tuple(self):
        return self.x, self.y


class Segment:
    EPSILON = 1e-10

    def __init__(self, pt0: Point2d, pt1: Point2d):
        self.pt0 = pt0
        self.pt1 = pt1

    @property
    def len(self):
        return math.hypot(self.pt0.x - self.pt1.x, self.pt0.y - self.pt1.y)

    def __mod__(self, other: "Segment"):
        # if (min(self.pt0.x, self.pt1.x) > max(other.pt0.x, other.pt1.x)
        #         or min(other.pt0.x, other.pt1.x) > max(self.pt0.x, self.pt1.x)
        #         or min(self.pt0.y, self.pt1.y) > max(other.pt0.y, other.pt1.y)
        #         or min(other.pt0.y, other.pt1.y) > max(self.pt0.y, self.pt1.y)):
        #     return self.pt1
        # if (
        #         (other.pt0.x - self.pt0.x) * (other.pt0.y - other.pt1.y)
        #         - (other.pt0.y - self.pt0.y) * (other.pt0.x - other.pt1.x)) * (
        #         (other.pt0.x - self.pt1.x) * (other.pt0.y - other.pt1.y)
        #         - (other.pt0.y - self.pt1.y) * (other.pt0.x - other.pt1.x)) >= 0 and (
        #         (self.pt0.x - other.pt0.x) * (self.pt0.y - self.pt1.y)
        #         - (self.pt0.y - other.pt0.y) * (self.pt0.x - self.pt1.x)) * (
        #         (self.pt0.x - other.pt1.x) * (self.pt0.y - self.pt1.y)
        #         - (self.pt0.y - other.pt1.y) * (self.pt0.x - self.pt1.x)) >= 0:
        #     return self.pt1
        d0 = self.pt1 - self.pt0
        d1 = other.pt1 - other.pt0
        diff = other.pt0 - self.pt0

        kross_d0_d1 = d0.x * d1.y - d1.x * d0.y
        len_d0 = d0.len
        len_d1 = d1.len
        if kross_d0_d1 > Segment.EPSILON * len_d0 * len_d1:
            s = (diff.x * d1.y - d1.x * diff.y) / kross_d0_d1
            if s < 0. or s > 1.:
                return self.pt1
            t = (diff.x * d0.y - d0.x * diff.y) / kross_d0_d1
            if t < 0. or t > 1.:
                return self.pt1
            return self.pt0 + d0 * s
        return self.pt1
        # len_diff = diff.len
        # kross_diff_d0 = diff.x * d0.y - d0.x * diff.y
        # if kross_diff_d0 > Segment.EPSILON * len_diff * len_d0:
        #     return self.pt1
        # delta_1 = other.pt0 - self.pt0
        # delta_2 = other.pt1 - self.pt0
        # if delta_1.len < delta_2.len:
        #     if delta_1.x * d0.x > 0 and delta_1.y * d0.y > 0:
        #         return other.pt0
        #     else:
        #         return self.pt1
        # else:
        #     if delta_2.x * d0.x > 0 and delta_2.y * d0.y > 0:
        #         return other.pt1
        #     else:
        #         return self.pt1


class Structure:
    def __init__(self, pts: list[Point2d]):
        self.pts = pts

    def __len__(self):
        return len(self.pts)

    @property
    def segments(self) -> list[Segment]:
        return [Segment(self.pts[i], self.pts[(i + 1) % len(self)]) for i in range(len(self))]


class Lidar:
    def __init__(self, lidar_rays: int, lidar_range: float, field_of_view: float):
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.field_of_view = field_of_view
        self.lidar_gap = field_of_view / (lidar_rays - 1)

    def detect(self,
               position: Point2d,
               direction: float,
               structures: list[Structure]) -> tuple[list[Point2d], np.ndarray]:
        direction_lb = -direction - self.field_of_view / 2
        pts_result = []
        len_result = np.zeros(shape=(self.lidar_rays,))
        for i in range(0, self.lidar_rays):
            beam_direction = direction_lb + i * self.lidar_gap
            point_dst = position + Point2d(
                x=math.cos(math.radians(beam_direction)),
                y=math.sin(math.radians(beam_direction)),
            ) * self.lidar_range
            beam_segment = Segment(position, point_dst)
            candidate = (point_dst, self.lidar_range)
            for structure in structures:
                for segment in structure.segments:
                    cross_pt = beam_segment % segment
                    cross_len = cross_pt % position
                    if cross_len < candidate[1]:
                        candidate = (cross_pt, cross_len)
            pts_result.append(candidate[0])
            len_result[i] = candidate[1]
        return pts_result, 1. - len_result / self.lidar_range
