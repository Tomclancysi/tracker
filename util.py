import cv2
import math
import numpy as np
import Polygon as poly
import os
import numpy as np

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # colinear
    return 1 if val > 0 else 2  # 1 for clockwise, 2 for counterclockwise

def graham_scan(points: list):
    n = len(points)
    if n < 3:
        return []

    # 找到具有最小y坐标的点（在相等的情况下，选择最左边的点）
    pivot = min(points, key=lambda point: (point[1], point[0]))

    # 对点进行极坐标排序
    sorted_points = sorted(points, key=lambda point: (math.atan2(point[1] - pivot[1], point[0] - pivot[0]), point))

    # 初始化凸包
    hull = [pivot, sorted_points[0], sorted_points[1]]

    for i in range(2, n):
        while len(hull) > 1 and orientation(hull[-2], hull[-1], sorted_points[i]) != 2:
            hull.pop()
        hull.append(sorted_points[i])

    return hull

def point_in_polygon(point, polygon):
    p = poly.Polygon(polygon)
    return p.isInside(*point)

def determine_overall_direction(vectors: np.array):
    """1表示从下往上，-1表示从上往下"""
    up_votes, down_votes = 0, 0
    for v in vectors:
        # [0, 1] X v
        if v[1] < 0:
            up_votes += 1
        elif v[1] > 0:
            down_votes += 1
    if up_votes > down_votes:
        return 1
    elif up_votes < down_votes:
        return -1
    return 0

class FrameReader:
    def __init__(self, path) -> None:
        self.path = path
        self.is_folder = os.path.isdir(path)
        self.frame_iterator = None

        if self.is_folder:
            self.frame_iterator = self.read_frames_from_folder()
        else:
            self.frame_iterator = self.read_frames_from_video()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.frame_iterator)

    def read_frames_from_folder(self):
        frame_number = 1
        while True:
            frame_path = os.path.join(self.path, f"{frame_number}.jpg")
            if not os.path.exists(frame_path):
                raise StopIteration
            frame = cv2.imread(frame_path)
            yield frame
            frame_number += 1

    def read_frames_from_video(self):
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise StopIteration
            yield frame


class Animation:
    def __init__(self, start_frame, duration) -> None:
        self.start_frame = start_frame
        self.duration = duration

    def animate(self, frame, canvases):
        pass
    
    def expired(self, frameNum):
        return frameNum > self.start_frame + self.duration


class LandingShadowAnimation(Animation):
    def __init__(self, start_frame, duration) -> None:
        super().__init__(start_frame, duration)
    
    def animate(self, frame, canvases):
        
        pass


class LandingPointAnimation(Animation):
    def __init__(self, start_frame, duration, x, y, color, frame_offset, radius, type) -> None:
        super().__init__(start_frame, duration)
        self.x, self.y, self.color, self.frame_offset, self.radius, self.type = x, y, color, frame_offset, radius, type
    
    def animate(self, frameNum, court, video):
        if frameNum < self.start_frame or frameNum > self.start_frame + self.duration:
            return
        curRadius = np.interp(frameNum - self.start_frame, self.frame_offset, self.radius)
        if self.type == 0:
            cv2.circle(court, (self.x, self.y), int(curRadius), self.color, -1)
        else:
            axes = (int(curRadius), int(curRadius*0.5))  # 长轴和短轴的长度
            cv2.ellipse(video, (self.x, self.y), axes, 0, 0, 360, self.color, -1)
