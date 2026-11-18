# 整合版：阈值调整 + 矩形检测
# 功能：
#   1. 阈值调整模式：触摸屏幕调整LAB二值化阈值，实时观察二值化效果
#   2. 矩形检测模式：检测图像中的最大矩形，使用卡尔曼滤波平滑坐标，并通过UART发送数据
# 操作说明：
#   - 屏幕右下角有“切换模式”按钮，点击可在两种模式间切换
#   - 在阈值调整模式下，左右两侧按钮用于调整阈值范围，中间十字点用于读取LAB值
#   - 在矩形检测模式下，自动检测矩形并输出数据

import time, math, os, gc, sys, urandom, image
from media.sensor import *
from media.display import *
from media.media import *
from machine import TOUCH
from ybUtils.YbUart import YbUart
from libs.YbProtocol import YbProtocol
import cv_lite

# ==================== 全局常量 ====================
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240
DETECT_WIDTH = 320          # 图像检测宽度
DETECT_HEIGHT = 240         # 图像检测高度
IMAGE_CENTER_X = DETECT_WIDTH // 2

# 颜色定义（用于阈值调整模式）
BUTTON_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 0)

# LAB阈值参数（初始范围）
lmin, lmax = 0, 100
amin, amax = -128, 127
bmin, bmax = -128, 127
threshold_set = (lmin, lmax, amin, amax, bmin, bmax)
red = (24, 48, 19, 75, -128, 13)  # 红色阈值示例（未使用）

# 模式标志
MODE_THRESHOLD = 0      # 阈值调整模式
MODE_RECT_DETECT = 1    # 矩形检测模式
current_mode = MODE_THRESHOLD

# 阈值调整模式内部标志
edit_flag = 0           # 0: 非编辑模式（显示原始图像+按钮）; 1: 编辑模式（显示二值化图像+调整按钮）

# 中心LAB值
l_center = 0
a_center = 0
b_center = 0

# 物体尺寸参数（矩形检测模式用）
KNOWN_WIDTH = 21.0      # 实际物体宽度(cm)
KNOWN_HEIGHT = 29.7     # 实际物体高度(cm)
FOCAL_LENGTH = 500      # 相机焦距(px)

# UART
uart1 = None
pto = YbProtocol()

# 触摸对象
tp = TOUCH(0)

# ==================== 卡尔曼滤波器类 ====================
class KalmanFilter:
    """二维卡尔曼滤波器，用于单个坐标点的跟踪"""
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.state = [0.0, 0.0, 0.0, 0.0]          # [x, y, vx, vy]
        self.P = [[1000.0, 0.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0, 0.0],
                  [0.0, 0.0, 1000.0, 0.0],
                  [0.0, 0.0, 0.0, 1000.0]]
        q = process_noise
        self.Q = [[q, 0.0, 0.0, 0.0],
                  [0.0, q, 0.0, 0.0],
                  [0.0, 0.0, q, 0.0],
                  [0.0, 0.0, 0.0, q]]
        r = measurement_noise
        self.R = [[r, 0.0],
                  [0.0, r]]
        self.H = [[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]]
        self.initialized = False

    def predict(self, dt=1.0):
        if not self.initialized:
            return
        F = [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        new_state = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            for j in range(4):
                new_state[i] += F[i][j] * self.state[j]
        self.state = new_state
        FP = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    FP[i][j] += F[i][k] * self.P[k][j]
        FPFT = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    FPFT[i][j] += FP[i][k] * F[j][k]
        for i in range(4):
            for j in range(4):
                self.P[i][j] = FPFT[i][j] + self.Q[i][j]

    def update(self, measurement):
        x, y = measurement
        if not self.initialized:
            self.state = [x, y, 0.0, 0.0]
            self.initialized = True
            return
        predicted_measurement = [self.state[0], self.state[1]]
        residual = [x - predicted_measurement[0], y - predicted_measurement[1]]
        HP = [[0.0] * 4 for _ in range(2)]
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    HP[i][j] += self.H[i][k] * self.P[k][j]
        HPHT = [[0.0] * 2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    HPHT[i][j] += HP[i][k] * self.H[j][k]
        S = [[HPHT[i][j] + self.R[i][j] for j in range(2)] for i in range(2)]
        det_S = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        if abs(det_S) < 1e-10:
            return
        S_inv = [[S[1][1] / det_S, -S[0][1] / det_S],
                 [-S[1][0] / det_S, S[0][0] / det_S]]
        PHT = [[0.0] * 2 for _ in range(4)]
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    PHT[i][j] += self.P[i][k] * self.H[j][k]
        K = [[0.0] * 2 for _ in range(4)]
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    K[i][j] += PHT[i][k] * S_inv[k][j]
        for i in range(4):
            for j in range(2):
                self.state[i] += K[i][j] * residual[j]
        KH = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    KH[i][j] += K[i][k] * self.H[k][j]
        I_KH = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                I_KH[i][j] = (1.0 if i == j else 0.0) - KH[i][j]
        new_P = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    new_P[i][j] += I_KH[i][k] * self.P[k][j]
        self.P = new_P

    def get_position(self):
        if not self.initialized:
            return None
        return (int(self.state[0]), int(self.state[1]))

    def reset(self):
        self.initialized = False
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.P = [[1000.0, 0.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0, 0.0],
                  [0.0, 0.0, 1000.0, 0.0],
                  [0.0, 0.0, 0.0, 1000.0]]

class CoordinateFilter:
    """坐标滤波器，使用卡尔曼滤波平滑矩形坐标变化"""
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.corner_filters = [KalmanFilter(process_noise, measurement_noise) for _ in range(4)]
        self.center_filter = KalmanFilter(process_noise, measurement_noise)
        self.dt = 1.0
        self.data_count = 0

    def add_corners(self, corners):
        self.data_count += 1
        for i, corner in enumerate(corners):
            self.corner_filters[i].predict(self.dt)
            self.corner_filters[i].update(corner)

    def add_center(self, center):
        self.center_filter.predict(self.dt)
        self.center_filter.update(center)

    def get_filtered_corners(self):
        if self.data_count < 2:
            return None
        filtered_corners = []
        for corner_filter in self.corner_filters:
            pos = corner_filter.get_position()
            if pos is None:
                return None
            filtered_corners.append(pos)
        return filtered_corners

    def get_filtered_center(self):
        if self.data_count < 2:
            return None
        return self.center_filter.get_position()

    def reset(self):
        for corner_filter in self.corner_filters:
            corner_filter.reset()
        self.center_filter.reset()
        self.data_count = 0

# ==================== 辅助函数 ====================
def rgb_to_lab(r, g, b):
    """RGB转LAB颜色空间"""
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    r = r ** 2.2 if r > 0.04045 else r / 12.92
    g = g ** 2.2 if g > 0.04045 else g / 12.92
    b = b ** 2.2 if b > 0.04045 else b / 12.92
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    x /= 0.95047
    y /= 1.0
    z /= 1.08883
    x = x ** (1/3) if x > 0.008856 else 7.787 * x + 16/116
    y = y ** (1/3) if y > 0.008856 else 7.787 * y + 16/116
    z = z ** (1/3) if z > 0.008856 else 7.787 * z + 16/116
    l = max(0, min(100, 116 * y - 16))
    a = max(-128, min(127, 500 * (x - y)))
    b_val = max(-128, min(127, 200 * (y - z)))
    return (int(l), int(a), int(b_val))

def get_center_lab(img, x, y):
    """获取指定点的LAB值"""
    x = max(0, min(DISPLAY_WIDTH-1, x))
    y = max(0, min(DISPLAY_HEIGHT-1, y))
    r, g, b = img.get_pixel(x, y)
    return rgb_to_lab(r, g, b)

def calculate_center(corners):
    """计算矩形四个角点的对角线交点作为中心点坐标"""
    if len(corners) != 4:
        return None
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    x4, y4 = corners[3]
    try:
        dx1 = x3 - x1
        dy1 = y3 - y1
        dx2 = x4 - x2
        dy2 = y4 - y2
        det = dx1 * (-dy2) - dy1 * (-dx2)
        if abs(det) < 1e-10:
            cx = sum(corner[0] for corner in corners) / 4
            cy = sum(corner[1] for corner in corners) / 4
        else:
            t1 = ((x2 - x1) * (-dy2) - (y2 - y1) * (-dx2)) / det
            cx = x1 + t1 * dx1
            cy = y1 + t1 * dy1
    except:
        cx = sum(corner[0] for corner in corners) / 4
        cy = sum(corner[1] for corner in corners) / 4
    return (int(cx), int(cy))

def init_sensor():
    """初始化摄像头"""
    sensor = Sensor(width=DETECT_WIDTH, height=DETECT_HEIGHT)
    sensor.reset()
    sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)
    return sensor

def init_display():
    """初始化显示"""
    Display.init(Display.ST7701, to_ide=True)
    MediaManager.init()

def deinit_display():
    """释放显示资源"""
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(50)
    Display.deinit()
    MediaManager.deinit()

def uart_init():
    """初始化UART"""
    global uart1
    uart1 = YbUart(baudrate=115200)

def uart_deinit():
    global uart1
    if uart1:
        uart1.deinit()
        uart1 = None

# ==================== 阈值调整模式 ====================
def draw_threshold_buttons(img):
    """绘制阈值调整模式的按钮"""
    # 左侧按钮（最小值调节和复位）
    img.draw_rectangle(0, 0, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False)   # 返回（此处用于切换模式）
    img.draw_string_advanced(10, 10, 40, "模式切换", color=TEXT_COLOR)
    img.draw_rectangle(0, 100, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False) # Lmin+
    img.draw_string_advanced(10, 110, 40, "Lmin+", color=TEXT_COLOR)
    img.draw_rectangle(0, 200, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False) # Amin+
    img.draw_string_advanced(10, 210, 40, "Amin+", color=TEXT_COLOR)
    img.draw_rectangle(0, 300, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False) # Bmin+
    img.draw_string_advanced(10, 310, 40, "Bmin+", color=TEXT_COLOR)
    img.draw_rectangle(0, 400, 160, 80, color=BUTTON_COLOR, thickness=2, fill=False)  # 复位
    img.draw_string_advanced(10, 410, 40, "复位", color=TEXT_COLOR)

    # 右侧按钮（最大值调节）
    img.draw_rectangle(480, 100, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False) # Lmax-
    img.draw_string_advanced(490, 110, 40, "Lmax-", color=TEXT_COLOR)
    img.draw_rectangle(480, 200, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False) # Amax-
    img.draw_string_advanced(490, 210, 40, "Amax-", color=TEXT_COLOR)
    img.draw_rectangle(480, 300, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False) # Bmax-
    img.draw_string_advanced(490, 310, 40, "Bmax-", color=TEXT_COLOR)

    # 显示当前阈值
    threshold_text = f"({lmin},{lmax}, {amin},{amax}, {bmin},{bmax})"
    img.draw_string_advanced(300, 20, 30, threshold_text, color=BUTTON_COLOR)

def edit_threshold_mode(img):
    """编辑模式下的按钮绘制（编辑按钮）"""
    global edit_flag
    # 绘制"编辑"按钮（位于右上角）
    img.draw_rectangle(480, 0, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False)
    img.draw_string_advanced(490, 10, 40, "编辑", color=TEXT_COLOR)
    # 检测触摸切换编辑模式
    touch_points = tp.read(1)
    if touch_points:
        x, y = touch_points[0].x, touch_points[0].y
        if 480 <= x <= 640 and 0 <= y <= 100:
            edit_flag = 1

def read_center_lab(img):
    """读取中心LAB值"""
    global l_center, a_center, b_center
    # 绘制"读取"按钮
    img.draw_rectangle(480, 100, 160, 100, color=BUTTON_COLOR, thickness=2, fill=False)
    img.draw_string_advanced(490, 110, 40, "读取", color=TEXT_COLOR)
    # 绘制中心十字线
    img.draw_line(310, 240, 319, 240, BUTTON_COLOR)
    img.draw_line(321, 240, 330, 240, BUTTON_COLOR)
    img.draw_line(320, 230, 320, 239, BUTTON_COLOR)
    img.draw_line(320, 241, 320, 250, BUTTON_COLOR)
    # 检测触摸读取值
    touch_points = tp.read(1)
    if touch_points:
        x, y = touch_points[0].x, touch_points[0].y
        if 480 <= x <= 640 and 100 <= y <= 200:
            l_center, a_center, b_center = get_center_lab(img, 320, 240)
    # 显示LAB值
    lab_text = f"LAB: L={l_center}, A={a_center}, B={b_center}"
    img.draw_string_advanced(320, 450, 30, lab_text, color=TEXT_COLOR)

def check_threshold_touch():
    """检测触摸并更新阈值"""
    global lmin, lmax, amin, amax, bmin, bmax, edit_flag, threshold_set
    touch_points = tp.read(1)
    if touch_points:
        x, y = touch_points[0].x, touch_points[0].y
        # 左侧按钮区域
        if 0 <= x <= 160:
            if 100 <= y <= 200:          # Lmin+
                lmin = min(lmin + 2, lmax)
            elif 200 <= y <= 300:        # Amin+
                amin = min(amin + 2, amax)
            elif 300 <= y <= 400:        # Bmin+
                bmin = min(bmin + 2, bmax)
            elif 0 <= y <= 100:          # 模式切换
                global current_mode
                current_mode = MODE_RECT_DETECT
                edit_flag = 0             # 重置编辑标志
                return
            elif 400 <= y <= 480:        # 复位
                lmin, lmax = 0, 100
                amin, amax = -128, 127
                bmin, bmax = -128, 127
        # 右侧按钮区域
        elif 480 <= x <= 640:
            if 100 <= y <= 200:          # Lmax-
                lmax = max(lmax - 2, lmin)
            elif 200 <= y <= 300:        # Amax-
                amax = max(amax - 2, amin)
            elif 300 <= y <= 400:        # Bmax-
                bmax = max(bmax - 2, bmin)
        # 更新阈值集合
        threshold_set = (lmin, lmax, amin, amax, bmin, bmax)
        time.sleep_ms(200)  # 防抖

def threshold_mode_loop(img):
    """阈值调整模式主逻辑（单帧处理）"""
    global edit_flag
    if edit_flag == 0:
        # 非编辑模式：显示原始图像，绘制按钮，读取LAB值
        edit_threshold_mode(img)
        read_center_lab(img)
    else:
        # 编辑模式：二值化图像，绘制调整按钮，处理触摸事件
        img.binary([threshold_set], invert=False)
        draw_threshold_buttons(img)
        check_threshold_touch()
        # 显示LAB值
        lab_text = f"LAB: L={l_center}, A={a_center}, B={b_center}"
        img.draw_string_advanced(320, 450, 30, lab_text, color=TEXT_COLOR)

# ==================== 矩形检测模式 ====================
# 全局滤波器实例
coord_filter = CoordinateFilter(process_noise=1e-2, measurement_noise=1e-1)

def draw_mode_switch_button(img, mode_text):
    """在屏幕右下角绘制模式切换按钮（通用）"""
    btn_x = DISPLAY_WIDTH - 100
    btn_y = DISPLAY_HEIGHT - 50
    img.draw_rectangle(btn_x, btn_y, 100, 50, color=(0, 255, 0), thickness=2, fill=False)
    img.draw_string_advanced(btn_x + 10, btn_y + 10, 30, mode_text, color=(0, 255, 0))

def rect_detect_mode_loop(img):
    """矩形检测模式主逻辑（单帧处理）"""
    # 绘制模式切换按钮（位于右下角）
    draw_mode_switch_button(img, "阈值模式")
    # 检测触摸切换模式
    touch_points = tp.read(1)
    if touch_points:
        x, y = touch_points[0].x, touch_points[0].y
        btn_x = DISPLAY_WIDTH - 100
        btn_y = DISPLAY_HEIGHT - 50
        if btn_x <= x <= btn_x+100 and btn_y <= y <= btn_y+50:
            global current_mode
            current_mode = MODE_THRESHOLD
            coord_filter.reset()   # 重置滤波器
            return

    # 转为灰度图
    img_gray = img.to_grayscale()
    # 二值化（OTSU）
    hist = img_gray.get_histogram()
    otsu_threshold_obj = hist.get_threshold()
    threshold_value = (otsu_threshold_obj.value(), 255)
    img_binary = img_gray.binary([threshold_value])
    # 获取灰度numpy数组用于cv_lite
    img_gray_np = img_gray.to_numpy_ref()
    image_shape = [DETECT_HEIGHT, DETECT_WIDTH]

    # 矩形检测参数
    canny_thresh1 = 50
    canny_thresh2 = 150
    approx_epsilon = 0.04
    area_min_ratio = 0.01
    max_angle_cos = 0.3
    gaussian_blur_size = 5

    rects_data = cv_lite.grayscale_find_rectangles(
        image_shape, img_gray_np,
        canny_thresh1, canny_thresh2,
        approx_epsilon,
        area_min_ratio,
        max_angle_cos,
        gaussian_blur_size
    )

    max_rect = None
    x_error = 0.0
    if rects_data and len(rects_data) >= 4:
        filtered_rects = []
        for i in range(0, len(rects_data), 4):
            if i + 3 < len(rects_data):
                x, y, w, h = rects_data[i], rects_data[i+1], rects_data[i+2], rects_data[i+3]
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                if area > 1500 and 0.6 < aspect_ratio < 1.6:
                    filtered_rects.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
        if filtered_rects:
            max_rect = max(filtered_rects, key=lambda r: r['area'])
            if max_rect:
                x, y, w, h = max_rect['x'], max_rect['y'], max_rect['w'], max_rect['h']
                corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                center = calculate_center(corners)
                x_error = center[0] - IMAGE_CENTER_X
                # 限制误差范围
                x_error = max(-100, min(100, x_error))
                # 添加到滤波器
                coord_filter.add_corners(corners)
                coord_filter.add_center(center)
                # 获取滤波后的坐标
                filtered_corners = coord_filter.get_filtered_corners()
                filtered_center = coord_filter.get_filtered_center()
                if filtered_corners and filtered_center:
                    min_x = min(c[0] for c in filtered_corners)
                    max_x = max(c[0] for c in filtered_corners)
                    min_y = min(c[1] for c in filtered_corners)
                    max_y = max(c[1] for c in filtered_corners)
                    img.draw_rectangle([min_x, min_y, max_x-min_x, max_y-min_y], color=(0,255,0), thickness=2)
                    for corner in filtered_corners:
                        img.draw_circle(corner[0], corner[1], 5, color=(255,0,0), thickness=2)
                    img.draw_circle(filtered_center[0], filtered_center[1], 8, color=(0,0,255), thickness=2)
                else:
                    # 滤波器未稳定时绘制原始矩形
                    img.draw_rectangle([x, y, w, h], color=(0,255,0), thickness=2)
                    for corner in corners:
                        img.draw_circle(corner[0], corner[1], 5, color=(255,0,0), thickness=2)
                    img.draw_circle(center[0], center[1], 8, color=(0,0,255), thickness=2)
                # 通过UART发送数据
                if uart1:
                    try:
                        x_send = int(center[0])
                        y_send = int(center[1])
                        w_send = int(w)
                        h_send = int(h)
                        pto_data = pto.get_color_data(x_send, y_send, w_send, h_send)
                        uart1.send(pto_data)
                    except:
                        pass
            else:
                coord_filter.reset()
        else:
            coord_filter.reset()
    else:
        coord_filter.reset()

    # 显示FPS和模式信息
    # 注意：这里需要fps对象，我们在主循环中创建并传递
    # 改为在调用时由主循环传入fps值，或者直接在函数内部计算
    # 为了简单，这里仅绘制其他信息，FPS由主循环绘制
    img.draw_circle(IMAGE_CENTER_X, DETECT_HEIGHT//2, 3, color=(255,255,0), thickness=2)
    # 显示误差
    if 'x_error' in locals():
        error_text = f"Error: {x_error:.1f}"
        img.draw_string_advanced(DETECT_WIDTH-150, 5, 16, error_text, color=(0,255,255))

# ==================== 主函数 ====================
def main():
    global current_mode, edit_flag, sensor
    os.exitpoint(os.EXITPOINT_ENABLE)

    sensor = None
    try:
        # 初始化硬件
        sensor = init_sensor()
        init_display()
        uart_init()
        sensor.run()
        clock = time.clock()

        while True:
            clock.tick()
            img = sensor.snapshot()
            # 根据当前模式调用对应处理函数
            if current_mode == MODE_THRESHOLD:
                threshold_mode_loop(img)
            else:
                rect_detect_mode_loop(img)

            # 绘制通用信息（FPS）
            fps_text = f"FPS: {clock.fps():.1f}"
            mode_name = "Threshold" if current_mode == MODE_THRESHOLD else "Rect Detect"
            img.draw_string_advanced(5, 5, 16, fps_text, color=(255,255,255))
            img.draw_string_advanced(5, 25, 16, f"Mode: {mode_name}", color=(255,255,255))

            # 显示图像
            Display.show_image(img)
            gc.collect()

    except KeyboardInterrupt as e:
        print("用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if sensor:
            sensor.stop()
        deinit_display()
        uart_deinit()

if __name__ == "__main__":
    main()
