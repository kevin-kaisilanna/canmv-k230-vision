import time, os, sys, gc
import math
import cv_lite
import ulab.numpy as np
from media.sensor import *
from media.display import *
from media.media import *
from ybUtils.YbUart import YbUart
from libs.YbProtocol import YbProtocol

# ======================== 卡尔曼滤波器（2D点平滑） ========================
class KalmanFilter2D:
    """恒定速度模型卡尔曼滤波器，用于平滑单个坐标点 (x, y)"""
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = 1.0
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.P = [[1000.0, 0.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0, 0.0],
                  [0.0, 0.0, 1000.0, 0.0],
                  [0.0, 0.0, 0.0, 1000.0]]
        q = process_noise
        self.Q = [[q, 0, 0, 0],
                  [0, q, 0, 0],
                  [0, 0, q, 0],
                  [0, 0, 0, q]]
        self.H = [[1, 0, 0, 0],
                  [0, 1, 0, 0]]
        r = measurement_noise
        self.R = [[r, 0],
                  [0, r]]
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return
        F = [[1, 0, self.dt, 0],
             [0, 1, 0, self.dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        s = self.state
        self.state = [F[0][0]*s[0] + F[0][1]*s[1] + F[0][2]*s[2] + F[0][3]*s[3],
                      F[1][0]*s[0] + F[1][1]*s[1] + F[1][2]*s[2] + F[1][3]*s[3],
                      F[2][0]*s[0] + F[2][1]*s[1] + F[2][2]*s[2] + F[2][3]*s[3],
                      F[3][0]*s[0] + F[3][1]*s[1] + F[3][2]*s[2] + F[3][3]*s[3]]
        P = self.P
        FP = [[sum(F[i][k]*P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        FPFT = [[sum(FP[i][k]*F[j][k] for k in range(4)) for j in range(4)] for i in range(4)]
        self.P = [[FPFT[i][j] + self.Q[i][j] for j in range(4)] for i in range(4)]

    def update(self, z):
        x, y = z
        if not self.initialized:
            self.state = [x, y, 0.0, 0.0]
            self.initialized = True
            return
        hx = [self.state[0], self.state[1]]
        residual = [x - hx[0], y - hx[1]]
        HP = [[sum(self.H[i][k]*self.P[k][j] for k in range(4)) for j in range(4)] for i in range(2)]
        HPHT = [[sum(HP[i][k]*self.H[j][k] for k in range(4)) for j in range(2)] for i in range(2)]
        S = [[HPHT[i][j] + self.R[i][j] for j in range(2)] for i in range(2)]
        det = S[0][0]*S[1][1] - S[0][1]*S[1][0]
        if abs(det) < 1e-10:
            return
        S_inv = [[S[1][1]/det, -S[0][1]/det],
                 [-S[1][0]/det, S[0][0]/det]]
        PHT = [[sum(self.P[i][k]*self.H[j][k] for k in range(2)) for j in range(2)] for i in range(4)]
        K = [[sum(PHT[i][k]*S_inv[k][j] for k in range(2)) for j in range(2)] for i in range(4)]
        self.state = [self.state[i] + sum(K[i][j]*residual[j] for j in range(2)) for i in range(4)]
        I = [[1 if i==j else 0 for j in range(4)] for i in range(4)]
        KH = [[sum(K[i][k]*self.H[k][j] for k in range(2)) for j in range(4)] for i in range(4)]
        I_KH = [[I[i][j] - KH[i][j] for j in range(4)] for i in range(4)]
        self.P = [[sum(I_KH[i][k]*self.P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    def get_position(self):
        return (int(round(self.state[0])), int(round(self.state[1]))) if self.initialized else None

    def reset(self):
        self.initialized = False
        self.state = [0.0]*4
        self.P = [[1000.0, 0, 0, 0], [0, 1000.0, 0, 0], [0, 0, 1000.0, 0], [0, 0, 0, 1000.0]]

    # ---------- 新增：获取坐标和速度的估计误差（标准差） ----------
    def get_error(self):
        """
        返回当前位置(x,y)和速度(vx,vy)的标准差
        (如果滤波器未初始化则返回0)
        """
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0
        pos_err_x = math.sqrt(max(0, self.P[0][0]))
        pos_err_y = math.sqrt(max(0, self.P[1][1]))
        vel_err_x = math.sqrt(max(0, self.P[2][2]))
        vel_err_y = math.sqrt(max(0, self.P[3][3]))
        return pos_err_x, pos_err_y, vel_err_x, vel_err_y

# ======================== 硬件初始化 ========================
pto = YbProtocol()
uart1 = YbUart(baudrate=115200)
uart1.send("Hello kankanworld\n\r")

DETECT_WIDTH = 640
DETECT_HEIGHT = 480
IMAGE_CENTER_X = DETECT_WIDTH // 2
lcd_width = DETECT_WIDTH
lcd_height = DETECT_HEIGHT
DISPLAY_MODE = "LCD"

if DISPLAY_MODE == "VIRT":
    DISPLAY_WIDTH = DETECT_WIDTH
    DISPLAY_HEIGHT = DETECT_HEIGHT
elif DISPLAY_MODE == "LCD":
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
else:
    raise ValueError("Unknown DISPLAY_MODE")

sensor = None
sensor = Sensor(width=1920, height=1080)
sensor.reset()
sensor.set_framesize(width=320, height=240)
sensor.set_pixformat(Sensor.RGB565)

if DISPLAY_MODE == "VIRT":
    Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, fps=100, to_ide=True)
elif DISPLAY_MODE == "LCD":
    Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)

MediaManager.init()
sensor.run()

# ======================== 配置参数 ========================
canny_thresh1      = 50
canny_thresh2      = 150
approx_epsilon     = 0.04
area_min_ratio     = 0.005
max_angle_cos      = 0.3
gaussian_blur_size = 3

MIN_AREA = 500
MAX_AREA = 100000
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0

RECT_WIDTH = 210
RECT_HEIGHT = 95
BASE_RADIUS = 45
POINTS_PER_CIRCLE = 24
PURPLE_THRESHOLD = (20, 60, 15, 70, -70, -20)

CONFIRM_FRAMES = 1
LOST_FRAMES = 5

center_kf = KalmanFilter2D(process_noise=1e-3, measurement_noise=0.5)
stable_count = 0
lost_count = 0
last_sent_center = None

# ======================== 工具函数 ========================
def calculate_distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def calculate_center(points):
    if not points: return (0, 0)
    return (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))

def is_valid_rect(corners):
    edges = [calculate_distance(corners[i], corners[(i+1)%4]) for i in range(4)]
    ratio1 = edges[0] / max(edges[2], 0.1)
    ratio2 = edges[1] / max(edges[3], 0.1)
    valid_ratio = 0.5 < ratio1 < 1.5 and 0.5 < ratio2 < 1.5
    area = abs(sum(corners[i][0]*corners[(i+1)%4][1] - corners[(i+1)%4][0]*corners[i][1] for i in range(4))) / 2
    valid_area = MIN_AREA < area < MAX_AREA
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    w, h = max(xs)-min(xs), max(ys)-min(ys)
    valid_aspect = MIN_ASPECT_RATIO < w/max(h,0.1) < MAX_ASPECT_RATIO
    return valid_ratio and valid_area and valid_aspect

def sort_corners(corners):
    center = calculate_center(corners)
    sorted_pts = sorted(corners, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0]))
    if len(sorted_pts) == 4:
        left_top = min(sorted_pts, key=lambda p: p[0]+p[1])
        idx = sorted_pts.index(left_top)
        sorted_pts = sorted_pts[idx:] + sorted_pts[:idx]
    return sorted_pts

def get_perspective_matrix(src_pts, dst_pts):
    A = []; B = []
    for i in range(4):
        x,y = src_pts[i]; u,v = dst_pts[i]
        A.append([x,y,1,0,0,0,-u*x,-u*y])
        A.append([0,0,0,x,y,1,-v*x,-v*y])
        B.append(u); B.append(v)
    n = 8
    for i in range(n):
        pivot = i
        for j in range(i,n):
            if abs(A[j][i]) > abs(A[pivot][i]): pivot = j
        A[i], A[pivot] = A[pivot], A[i]
        B[i], B[pivot] = B[pivot], B[i]
        if abs(A[i][i]) < 1e-8: return None
        div = A[i][i]
        for j in range(i,n): A[i][j] /= div
        B[i] /= div
        for j in range(n):
            if j!=i and A[j][i]!=0:
                fac = A[j][i]
                for k in range(i,n): A[j][k] -= fac*A[i][k]
                B[j] -= fac*B[i]
    return [[B[0],B[1],B[2]], [B[3],B[4],B[5]], [B[6],B[7],1.0]]

def transform_points(points, matrix):
    res = []
    for x,y in points:
        xh = x*matrix[0][0] + y*matrix[0][1] + matrix[0][2]
        yh = x*matrix[1][0] + y*matrix[1][1] + matrix[1][2]
        wh = x*matrix[2][0] + y*matrix[2][1] + matrix[2][2]
        if abs(wh)>1e-8: res.append((xh/wh, yh/wh))
    return res

def detect_purple_blobs(img):
    return img.find_blobs([PURPLE_THRESHOLD], pixels_threshold=100, area_threshold=100, merge=True)


# ======================== 主循环 ========================
clock = time.clock()
image_shape = [sensor.height(), sensor.width()]  # 240, 320

while True:
    clock.tick()
    img = sensor.snapshot()

    # 1. 紫色色块检测
    purple_blobs = detect_purple_blobs(img)
    for blob in purple_blobs:
        img.draw_rectangle(blob[0:4], color=(255,0,255), thickness=1)
        img.draw_cross(blob.cx(), blob.cy(), color=(255,0,255), thickness=1)

    # 2. 矩形检测
    gray_img = img.to_grayscale()
    img_np = gray_img.to_numpy_ref()

    rects = cv_lite.grayscale_find_rectangles_with_corners(
        image_shape, img_np,
        canny_thresh1, canny_thresh2,
        approx_epsilon, area_min_ratio,
        max_angle_cos, gaussian_blur_size
    )

    # 3. 选择面积最大的有效矩形
    max_area = 0
    best_rect_corners = None

    for rect in rects:
        corners = [(rect[4], rect[5]), (rect[6], rect[7]),
                   (rect[8], rect[9]), (rect[10], rect[11])]
        if is_valid_rect(corners):
            area = rect[2] * rect[3]
            if area > max_area:
                max_area = area
                best_rect_corners = corners

    # 4. 稳定性处理与卡尔曼滤波
    if best_rect_corners is not None:
        sorted_corners = sort_corners(best_rect_corners)
        raw_center = calculate_center(sorted_corners)

        center_kf.predict()
        center_kf.update(raw_center)
        filtered_center = center_kf.get_position()

        stable_count += 1
        lost_count = 0

        if stable_count >= CONFIRM_FRAMES:
            # 绘制矩形
            for i in range(4):
                x1,y1 = sorted_corners[i]
                x2,y2 = sorted_corners[(i+1)%4]
                img.draw_line(x1,y1,x2,y2, color=(255,0,0), thickness=2)
            for p in sorted_corners:
                img.draw_circle(p[0], p[1], 5, color=(0,255,0), thickness=2)
            if filtered_center:
                img.draw_circle(filtered_center[0], filtered_center[1], 4, color=(0,255,255), thickness=2)

            # 透视变换与圆形采样点
            virtual_rect = [(0,0),(RECT_WIDTH,0),(RECT_WIDTH,RECT_HEIGHT),(0,RECT_HEIGHT)]
            matrix = get_perspective_matrix(virtual_rect, sorted_corners)
            if matrix:
                virtual_center = (RECT_WIDTH/2, RECT_HEIGHT/2)
                vir_pts = []
                for i in range(POINTS_PER_CIRCLE):
                    ang = 2*math.pi*i/POINTS_PER_CIRCLE
                    vir_pts.append((virtual_center[0]+BASE_RADIUS*math.cos(ang),
                                    virtual_center[1]+BASE_RADIUS*math.sin(ang)))
                mapped_pts = transform_points(vir_pts, matrix)
                for px, py in [(int(round(x)), int(round(y))) for x,y in mapped_pts]:
                    img.draw_circle(px, py, 2, color=(255,0,255), thickness=2)
                mapped_center = transform_points([virtual_center], matrix)
                if mapped_center:
                    mc = mapped_center[0]
                    img.draw_circle(int(round(mc[0])), int(round(mc[1])), 3, color=(0,0,255), thickness=1)

            # ---------- 串口发送（原有中心坐标 + 新增误差信息） ----------
            if filtered_center:
                x_send = filtered_center[0]
                y_send = filtered_center[1]
                # 原有发送保持不变
                try:
                    pto_data = pto.get_color_data(x_send, y_send, 0, 0)
                    uart1.send(pto_data)
                    last_sent_center = (x_send, y_send)
                    print(f"Sent rect: x={x_send}, y={y_send}, 0, 0")
                except Exception as e:
                    pass

                # 新增：发送误差信息（坐标标准差、速度标准差）
                pos_err_x, pos_err_y, vel_err_x, vel_err_y = center_kf.get_error()
                err_str = (f"ERR:{pos_err_x:.2f},{pos_err_y:.2f},",
                           f"{vel_err_x:.2f},{vel_err_y:.2f}\r\n")
                try:
                    uart1.send(err_str)
                except Exception:
                    pass
                print(err_str)   # 方便调试查看

    else:
        stable_count = 0
        lost_count += 1
        if lost_count >= LOST_FRAMES:
            center_kf.reset()
            last_sent_center = None

    # 绘制固定的摄像头中心十字
    cam_center_x = sensor.width() // 2
    cam_center_y = sensor.height() // 2
    img.draw_cross(cam_center_x, cam_center_y, color=(0, 255, 0), size=10, thickness=2)

    # 显示
    fps = clock.fps()
    img.draw_string_advanced(10, 10, 20, f"FPS: {fps:.1f}", color=(255, 255, 255))
    Display.show_image(img,
                      x=round((lcd_width - sensor.width()) / 2),
                      y=round((lcd_height - sensor.height()) / 2))
    print(f"FPS: {fps:.1f}")
    gc.collect()
