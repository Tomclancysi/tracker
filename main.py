import numpy as np
import cv2
import pandas as pd
import numpy.polynomial.polynomial as poly
import math
import util

from scipy.signal import savgol_filter

# Read Source Data
# INPUT_VIDEO = "./ss5_id_412.mp4"
# INPUT_LABEL = "./ss5_id_412.csv"
# OUTPUT_VIDEO = "./parabola.mp4"

INPUT_VIDEO = "D:\\ori_img"
INPUT_LABEL = "./8290/8290.csv"
OUTPUT_VIDEO = "./test.mp4"

imgSrc = util.FrameReader(INPUT_VIDEO)
df = pd.read_csv(INPUT_LABEL, encoding='gbk', index_col=None)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 24, (1280, 720))

x = df['x坐标']
y = df['y坐标']
hit = df['击球']
x = savgol_filter(x, 7, 3)
y = savgol_filter(y, 7, 3)

farend_color = np.array([255,178,0])
nearend_color = np.array([0,184,255])
eps = 0.0001
interval = 5
ball_width = 2

history = []
last_color = None

def distance(a, b): return (math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

def orientation(A, B, C):
    """
    判断点C在直线AB的左侧还是右侧
    返回值:
    1: 左侧
    -1: 右侧
    0: 直线上
    """
    cross_product = (B[1] - A[1]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[1] - A[1])
    if cross_product > 0:
        return 1  # 左侧
    elif cross_product < 0:
        return -1  # 右侧
    else:
        return 0  # 直线上

# https://stackoverflow.com/questions/57065080/draw-perpendicular-line-of-fixed-length-at-a-point-of-another-line
def perpendicular(slope, target, dist):
    dy = math.sqrt(3 ** 2 / (slope ** 2 + 1)) * dist
    dx = -slope * dy
    left = target[0] - dx, target[1] - dy
    right = target[0] + dx, target[1] + dy
    return left, right

def fill_hull_block(hull_points, history, img, color):
    bbox = [1e9, -1e9, 1e9, -1e9]
    for p in hull_points:
        bbox[0] = min(bbox[0], p[0])
        bbox[1] = max(bbox[1], p[0])
        bbox[2] = min(bbox[2], p[1])
        bbox[3] = max(bbox[3], p[1])
    m, n = bbox[3] - bbox[2] + 1, bbox[1] - bbox[0] + 1
    alpha = 1 / len(history)
    sq_dist = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    for i in range(m):
        for j in range(n):
            p = (j+bbox[0], i+bbox[2])
            if util.point_in_polygon(p, hull_points):
                k, dist = 0, sq_dist(p, history[0])
                for idx in range(1, len(history)):
                    t = sq_dist(p, history[idx])
                    if t < dist:
                        k, dist = idx, t
                t = alpha * (k+1)
                p = p[::-1]
                img[*p] = (img[*p].astype(float) * (1-t) + color.astype(float) * t).astype(np.uint8)
    return


for frameNum, img in enumerate(imgSrc):
    if frameNum < interval:
        continue
    if frameNum >= 2000:
        break
    smallDistance = 0.2

    # 当前时刻前后球的坐标，通过2阶多项式拟合
    if frameNum == 269:
        print("debug")
    x_values = x[frameNum - interval:frameNum + interval]
    if np.any(np.isclose(x_values, -1)):
        history = []
        last_color = None
    else:
        y_values = y[frameNum - interval:frameNum + interval]
        coefs = poly.polyfit(x_values, y_values, 2)

        # Calculate the points on either end of the tangent line
        target = int(x[frameNum]), int(poly.polyval(x[frameNum], coefs))
        leftX, rightX = x[frameNum] - smallDistance, x[frameNum] + smallDistance
        left = leftX, poly.polyval(leftX, coefs)
        right = rightX, poly.polyval(rightX, coefs)
        # Calculate the slope of the tangent line
        slope = (left[1] - right[1]) / (left[0] - right[0])  # tan(\theta)
        intercept = target[1] - slope * target[0]  # 直线和y轴交点的坐标y

        history.append(target)
        hull_points = [history[0]]
        dire_vec = []
        for i in range(0, len(history)-1):
            dx = history[i+1][0] - history[i][0]
            dy = history[i+1][1] - history[i][1]
            if dx == 0:
                dx = eps
            slopei = dy / dx
            dire_vec.append([dx, dy])
            width = ball_width / len(history) * (i+1)
            l, r = perpendicular(slopei, history[i+1], width)
            hull_points.append(l)
            hull_points.append(r)
        
        if len(hull_points) >= 3:
            try:
                hull_points = util.graham_scan(hull_points)
                hull_points = np.array(hull_points, dtype=np.int32)
                if last_color is None:
                    idx = util.determine_overall_direction(dire_vec) + 1
                    last_color = [farend_color, nearend_color, nearend_color][idx]
                else:
                    if hit[frameNum] == 1:
                        if y[frameNum] <= img.shape[0] / 2:
                            last_color = farend_color
                        else:
                            last_color = nearend_color
                fill_hull_block(hull_points, history, img, last_color)
                cv2.polylines(img, [hull_points], isClosed=True, color=(0, 255, 0), thickness=2)
            except:
                print("error in frame {}".format(frameNum))


        # Pop the oldest frame off the history if the history is longer than 0.25 seconds
        if len(history) > interval:
            history.pop(0)

    out.write(img)

    # Show Image
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if k == 27: break
    print(frameNum)
    frameNum += 1
cv2.destroyAllWindows()
