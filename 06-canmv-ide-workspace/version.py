# K230 寻找最大矩形框示例
# 功能：检测图像中的矩形，找到最大的矩形框，并打印四个角点和中心点坐标

import time, os, gc, sys, image
from media.sensor import *
from media.display import *
from media.media import *
# from machine import UART, Pin
from ybUtils.YbUart import YbUart
from libs.YbProtocol import YbProtocol
pto = YbProtocol()

import cv_lite

# 图像检测尺寸配置
DETECT_WIDTH = 640  # 宽度对齐到16的倍数
DETECT_HEIGHT = 480               # 高度
IMAGE_CENTER_X = DETECT_WIDTH // 2  # 图像中线x坐标

# 显示模式选择 Display mode selection
# 可选: "VIRT", "LCD"
# Options: "VIRT"(Virtual Display), "LCD"
DISPLAY_MODE = "LCD"

# 根据显示模式设置分辨率 Set resolution based on display mode
if DISPLAY_MODE == "VIRT":
    DISPLAY_WIDTH = DETECT_WIDTH
    DISPLAY_HEIGHT = DETECT_HEIGHT
elif DISPLAY_MODE == "LCD":
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
else:
    raise ValueError("Unknown DISPLAY_MODE, please select 'VIRT', 'LCD'")

sensor = None
uart1 = None  # UART1实例



import math

# 卡尔曼滤波器类
class KalmanFilter:
    """二维卡尔曼滤波器，用于单个坐标点的跟踪"""
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        # 状态向量 [x, y, vx, vy] - 位置和速度
        self.state = [0.0, 0.0, 0.0, 0.0]

        # 状态协方差矩阵 P (4x4)
        self.P = [[1000.0, 0.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0, 0.0],
                  [0.0, 0.0, 1000.0, 0.0],
                  [0.0, 0.0, 0.0, 1000.0]]

        # 过程噪声协方差矩阵 Q (4x4)
        q = process_noise
        self.Q = [[q, 0.0, 0.0, 0.0],
                  [0.0, q, 0.0, 0.0],
                  [0.0, 0.0, q, 0.0],
                  [0.0, 0.0, 0.0, q]]

        # 观测噪声协方差矩阵 R (2x2)
        r = measurement_noise
        self.R = [[r, 0.0],
                  [0.0, r]]

        # 观测矩阵 H (2x4) - 只观测位置
        self.H = [[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]]

        self.initialized = False

    def predict(self, dt=1.0):
        """预测步骤"""
        if not self.initialized:
            return

        # 状态转移矩阵 F (4x4)
        F = [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]

        # 预测状态: x = F * x
        new_state = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            for j in range(4):
                new_state[i] += F[i][j] * self.state[j]
        self.state = new_state

        # 预测协方差: P = F * P * F^T + Q
        # 计算 F * P
        FP = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    FP[i][j] += F[i][k] * self.P[k][j]

        # 计算 F * P * F^T
        FPFT = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    FPFT[i][j] += FP[i][k] * F[j][k]  # F^T[k][j] = F[j][k]

        # 添加过程噪声: P = F * P * F^T + Q
        for i in range(4):
            for j in range(4):
                self.P[i][j] = FPFT[i][j] + self.Q[i][j]

    def update(self, measurement):
        """更新步骤"""
        x, y = measurement

        if not self.initialized:
            # 初始化状态
            self.state = [x, y, 0.0, 0.0]
            self.initialized = True
            return

        # 计算观测残差: y = z - H * x
        predicted_measurement = [self.state[0], self.state[1]]  # H * x
        residual = [x - predicted_measurement[0], y - predicted_measurement[1]]

        # 计算残差协方差: S = H * P * H^T + R
        # H * P (2x4)
        HP = [[0.0] * 4 for _ in range(2)]
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    HP[i][j] += self.H[i][k] * self.P[k][j]

        # H * P * H^T (2x2)
        HPHT = [[0.0] * 2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    HPHT[i][j] += HP[i][k] * self.H[j][k]  # H^T[k][j] = H[j][k]

        # S = H * P * H^T + R
        S = [[HPHT[i][j] + self.R[i][j] for j in range(2)] for i in range(2)]

        # 计算卡尔曼增益: K = P * H^T * S^(-1)
        # 计算 S 的逆矩阵 (2x2)
        det_S = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        if abs(det_S) < 1e-10:
            return  # 矩阵奇异，跳过更新

        S_inv = [[S[1][1] / det_S, -S[0][1] / det_S],
                 [-S[1][0] / det_S, S[0][0] / det_S]]

        # P * H^T (4x2)
        PHT = [[0.0] * 2 for _ in range(4)]
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    PHT[i][j] += self.P[i][k] * self.H[j][k]  # H^T[k][j] = H[j][k]

        # K = P * H^T * S^(-1) (4x2)
        K = [[0.0] * 2 for _ in range(4)]
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    K[i][j] += PHT[i][k] * S_inv[k][j]

        # 更新状态: x = x + K * y
        for i in range(4):
            for j in range(2):
                self.state[i] += K[i][j] * residual[j]

        # 更新协方差: P = (I - K * H) * P
        # 计算 K * H (4x4)
        KH = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    KH[i][j] += K[i][k] * self.H[k][j]

        # 计算 (I - K * H)
        I_KH = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                I_KH[i][j] = (1.0 if i == j else 0.0) - KH[i][j]

        # P = (I - K * H) * P
        new_P = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    new_P[i][j] += I_KH[i][k] * self.P[k][j]
        self.P = new_P

    def get_position(self):
        """获取当前位置估计"""
        if not self.initialized:
            return None
        return (int(self.state[0]), int(self.state[1]))

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.state = [0.0, 0.0, 0.0, 0.0]
        # 重置协方差矩阵
        self.P = [[1000.0, 0.0, 0.0, 0.0],
                  [0.0, 1000.0, 0.0, 0.0],
                  [0.0, 0.0, 1000.0, 0.0],
                  [0.0, 0.0, 0.0, 1000.0]]

# 坐标滤波器类（使用卡尔曼滤波）
class CoordinateFilter:
    """坐标滤波器，使用卡尔曼滤波平滑矩形坐标变化"""
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        # 为四个角点和中心点分别创建卡尔曼滤波器
        self.corner_filters = [KalmanFilter(process_noise, measurement_noise) for _ in range(4)]
        self.center_filter = KalmanFilter(process_noise, measurement_noise)
        self.dt = 1.0  # 时间步长
        self.data_count = 0  # 数据计数器

    def add_corners(self, corners):
        """添加新的角点坐标并进行卡尔曼滤波"""
        self.data_count += 1

        for i, corner in enumerate(corners):
            # 预测步骤
            self.corner_filters[i].predict(self.dt)
            # 更新步骤
            self.corner_filters[i].update(corner)

    def add_center(self, center):
        """添加新的中心点坐标并进行卡尔曼滤波"""
        # 预测步骤
        self.center_filter.predict(self.dt)
        # 更新步骤
        self.center_filter.update(center)

    def get_filtered_corners(self):
        """获取卡尔曼滤波后的角点坐标"""
        if self.data_count < 2:  # 需要至少2帧数据才能提供稳定的滤波结果
            return None

        filtered_corners = []
        for corner_filter in self.corner_filters:
            pos = corner_filter.get_position()
            if pos is None:
                return None
            filtered_corners.append(pos)
        return filtered_corners

    def get_filtered_center(self):
        """获取卡尔曼滤波后的中心点坐标"""
        if self.data_count < 2:  # 需要至少2帧数据才能提供稳定的滤波结果
            return None
        return self.center_filter.get_position()

    def reset(self):
        """重置所有卡尔曼滤波器"""
        for corner_filter in self.corner_filters:
            corner_filter.reset()
        self.center_filter.reset()
        self.data_count = 0



# 创建全局滤波器实例（使用卡尔曼滤波）
coord_filter = CoordinateFilter(process_noise=1e-2, measurement_noise=1e-1)

def uart_init():
    """初始化UART串口"""
    global uart1

    # 配置GPIO9为UART1_TXD，GPIO10为UART1_RXD
    # 创建UART1实例，设置波特率为115200
    uart1 = YbUart(baudrate=115200)

    # 发送测试消息
    uart1.send("Hello kankanworld\n\r")

def camera_init():
    """初始化摄像头"""
    global sensor

    # 使用默认配置构造传感器对象
    sensor = Sensor(width=DETECT_WIDTH, height=DETECT_HEIGHT)
    # 重置传感器
    sensor.reset()
    # 设置水平镜像（可选）
    # sensor.set_hmirror(False)
    # 设置垂直翻转（可选）
    # sensor.set_vflip(False)

    # 设置通道0输出尺寸
    sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT)
    # 设置通道0输出格式为RGB565
    sensor.set_pixformat(Sensor.RGB565)

    # 根据显示模式初始化显示器 Initialize display based on display mode
    if DISPLAY_MODE == "VIRT":
        # 使用IDE作为显示输出 Use IDE as display output
        Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, fps=100, to_ide=True)
    elif DISPLAY_MODE == "LCD":
        # 使用LCD屏幕显示 Use LCD screen display
        Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)

    # 初始化媒体管理器
    MediaManager.init()
    # 启动传感器运行
    sensor.run()

    # print(f"显示模式: {DISPLAY_MODE}, 显示分辨率: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    # print(f"图像分辨率: {DETECT_WIDTH}x{DETECT_HEIGHT}")

def uart_deinit():
    """释放UART串口资源"""
    global uart1
    if uart1:
        uart1.deinit()
        uart1 = None
        # print("UART1资源已释放")

def camera_deinit():
    """释放摄像头资源"""
    global sensor
    # 停止传感器运行
    sensor.stop()
    # 释放显示资源
    Display.deinit()
    # 启用睡眠模式
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    # 释放媒体缓冲区
    MediaManager.deinit()

def set_display_mode(mode):
    """设置显示模式

    Args:
        mode (str): 显示模式，可选 "VIRT" 或 "LCD"
    """
    global DISPLAY_MODE, DISPLAY_WIDTH, DISPLAY_HEIGHT

    if mode in ["VIRT", "LCD"]:
        DISPLAY_MODE = mode
        if DISPLAY_MODE == "VIRT":
            DISPLAY_WIDTH = DETECT_WIDTH
            DISPLAY_HEIGHT = DETECT_HEIGHT
        elif DISPLAY_MODE == "LCD":
            DISPLAY_WIDTH = 640
            DISPLAY_HEIGHT = 480
        # print(f"显示模式已设置为: {DISPLAY_MODE}")
        # print(f"显示分辨率: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    else:
        # print(f"无效的显示模式: {mode}，请选择 'VIRT' 或 'LCD'")
        pass

def calculate_center(corners):
    """计算矩形四个角点的两条对角线交点作为中心点坐标

    Args:
        corners: 四个角点的坐标列表，格式为[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
        tuple: 对角线交点坐标(cx, cy)
    """
    if len(corners) != 4:
        return None

    # 提取四个角点坐标
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    x4, y4 = corners[3]

    try:
        # 方法1：计算两条对角线的交点
        # 对角线1：从第一个点到第三个点 (x1,y1) -> (x3,y3)
        # 对角线2：从第二个点到第四个点 (x2,y2) -> (x4,y4)

        # 计算对角线1的参数方程：P1 = (x1,y1) + t1*((x3,y3)-(x1,y1))
        # 计算对角线2的参数方程：P2 = (x2,y2) + t2*((x4,y4)-(x2,y2))

        # 对角线1的方向向量
        dx1 = x3 - x1
        dy1 = y3 - y1

        # 对角线2的方向向量
        dx2 = x4 - x2
        dy2 = y4 - y2

        # 求解交点：x1 + t1*dx1 = x2 + t2*dx2, y1 + t1*dy1 = y2 + t2*dy2
        # 整理得：t1*dx1 - t2*dx2 = x2 - x1, t1*dy1 - t2*dy2 = y2 - y1

        # 使用克拉默法则求解线性方程组
        det = dx1 * (-dy2) - dy1 * (-dx2)  # 行列式

        if abs(det) < 1e-10:  # 对角线平行或重合，使用几何中心
            cx = sum(corner[0] for corner in corners) / 4
            cy = sum(corner[1] for corner in corners) / 4
        else:
            # 计算参数t1
            t1 = ((x2 - x1) * (-dy2) - (y2 - y1) * (-dx2)) / det

            # 计算交点坐标
            cx = x1 + t1 * dx1
            cy = y1 + t1 * dy1

    except (ZeroDivisionError, ValueError):
        # 如果计算失败，回退到几何中心
        cx = sum(corner[0] for corner in corners) / 4
        cy = sum(corner[1] for corner in corners) / 4

    return (int(cx), int(cy))





def capture_picture():
    """捕获图像并进行矩形检测"""
    fps = time.clock()

    # 初始化变量，避免未定义错误
    max_rect = None
    x_error = 0.0
    pid_output = 0.0
    pid_abs = 0.0
    pid_scaled = 0.0
    filtered_corners = None
    filtered_center = None

    while True:
        fps.tick()
        try:
            os.exitpoint()
            global sensor
            img = sensor.snapshot()

            # 针对黑色边框白色内部的矩形进行检测
            # 先转换为灰度图像
            img_gray = img.to_grayscale()

            # 使用二值化处理，突出黑白对比
            # 使用固定阈值进行二值化
            # img_binary = img_gray.binary([0, 128])
            # 方法1: 使用OTSU自适应阈值（推荐）
            # 计算图像直方图并获取OTSU最佳阈值
            hist = img_gray.get_histogram()
            otsu_threshold_obj = hist.get_threshold()
            # 从threshold对象中提取阈值范围
            threshold_value = (otsu_threshold_obj.value(), 255)
            # 使用OTSU阈值进行二值化
            img_binary = img_gray.binary([threshold_value])

            # 方法3: 使用反向二值化，突出黑色边框
            # img_binary = img_gray.binary([(0, 100)], invert=True)

            # 使用cv_lite进行矩形检测
            image_shape = [DETECT_HEIGHT, DETECT_WIDTH]  # 高，宽
            img_gray_np = img_gray.to_numpy_ref()  # 获取灰度图像的numpy引用

            # cv_lite矩形检测参数
            canny_thresh1 = 50        # Canny 边缘检测低阈值
            canny_thresh2 = 150       # Canny 边缘检测高阈值
            approx_epsilon = 0.04     # 多边形拟合精度比例
            area_min_ratio = 0.01     # 最小面积比例（相对于图像总面积）
            max_angle_cos = 0.3       # 最大角度余弦（越小越接近矩形）
            gaussian_blur_size = 5    # 高斯模糊核尺寸

            # 调用cv_lite进行矩形检测
            rects_data = cv_lite.grayscale_find_rectangles(
                image_shape, img_gray_np,
                canny_thresh1, canny_thresh2,
                approx_epsilon,
                area_min_ratio,
                max_angle_cos,
                gaussian_blur_size
            )

            # 处理检测结果
            if rects_data and len(rects_data) >= 4:
                # 矩形过滤处理
                filtered_rects = []

                # 将返回的数据按每4个元素分组为矩形 [x, y, w, h]
                for i in range(0, len(rects_data), 4):
                    if i + 3 < len(rects_data):
                        x, y, w, h = rects_data[i], rects_data[i+1], rects_data[i+2], rects_data[i+3]

                        # 面积筛选
                        area = w * h

                        # 宽高比过滤：限制宽高比在合理范围内
                        aspect_ratio = w / h if h > 0 else 0
                        img.draw_string_advanced(5, 45, 16, f"Area: {aspect_ratio:.2f}", color=(0, 255, 255, 0))

                        # 过滤条件：面积大于最小阈值且宽高比在合理范围内
                        if area > 1500 and 0.6 < aspect_ratio < 1.6:
                            # 创建矩形对象用于后续处理
                            rect_info = {
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'area': area, 'aspect_ratio': aspect_ratio
                            }
                            filtered_rects.append(rect_info)

                # 最大面积选择：选择满足条件的最大面积矩形
                if filtered_rects:
                    max_rect = max(filtered_rects, key=lambda r: r['area'])
                else:
                    max_rect = None

                if max_rect:
                    # 从cv_lite检测结果计算矩形的四个角点
                    x, y, w, h = max_rect['x'], max_rect['y'], max_rect['w'], max_rect['h']
                    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                    # 计算中心点
                    center = calculate_center(corners)

                    # 计算中心x坐标与图像中线的误差
                    x_error = center[0] - IMAGE_CENTER_X

                    if x_error > 100:
                        x_error = 100
                    elif x_error < -100:
                        x_error = -100

                    # 将当前坐标添加到滤波器
                    global coord_filter
                    coord_filter.add_corners(corners)
                    coord_filter.add_center(center)

                    # 通过串口直接发送误差值
#                    if uart1:
#                        try:
#                            # 构造串口发送的数据帧
#                            # 帧格式: 帧头(0x66 0x66) + 误差值(2字节,有符号) + 帧尾(0xf6 0xf6)
#                            error_int = int(x_error)  # 转换为整数，保持原有精度

#                            # 将有符号整数转换为2字节的小端序字节数组
#                            if error_int < 0:
#                                # 负数使用补码表示
#                                error_bytes = (error_int + 65536).to_bytes(2, 'little')
#                            else:
#                                # 正数直接转换
#                                error_bytes = error_int.to_bytes(2, 'little')

#                            frame_data = bytes([0x66, 0x66]) + error_bytes + bytes([0xf6, 0xf6])

#                            uart1.write(frame_data)
                    if uart1:
                        try:
                            # K230 → STM32 标准协议：发送矩形位置数据
                            x_send = int(center[0])
                            y_send = int(center[1])
                            w_send = int(max_rect['w'])
                            h_send = int(max_rect['h'])

                            pto_data = pto.get_color_data(x_send, y_send, w_send, h_send)
                            uart1.send(pto_data)

                        except Exception as e:
                            # 串口发送异常时不中断主程序运行
                            pass

                    # 获取滤波后的坐标
                    filtered_corners = coord_filter.get_filtered_corners()
                    filtered_center = coord_filter.get_filtered_center()
                else:
                    # 没有找到符合条件的矩形，重置滤波器或保持上一次的状态
                    # print("未找到符合条件的矩形（面积>5000且宽高比1-2之间）")
                    filtered_corners = coord_filter.get_filtered_corners()
                    filtered_center = coord_filter.get_filtered_center()

                # 使用滤波后的坐标在二值化图像上进行绘制
                if filtered_corners and filtered_center:
                    # 计算滤波后的矩形边界框
                    min_x = min(corner[0] for corner in filtered_corners)
                    max_x = max(corner[0] for corner in filtered_corners)
                    min_y = min(corner[1] for corner in filtered_corners)
                    max_y = max(corner[1] for corner in filtered_corners)

                    # 在原图上绘制滤波后的矩形框
                    img.draw_rectangle([min_x, min_y, max_x - min_x, max_y - min_y], color=(0, 255, 0), thickness=2)

                    # 在原图上绘制滤波后的四个角点
                    for i, corner in enumerate(filtered_corners):
                        img.draw_circle(corner[0], corner[1], 5, color=(255, 0, 0), thickness=2)
                        # 在角点旁边标注序号
                        # img.draw_string_advanced(corner[0] + 8, corner[1] - 8, 16, str(i+1), color=(255, 255, 255, 0))

                    # 在原图上绘制滤波后的中心点
                    img.draw_circle(filtered_center[0], filtered_center[1], 8, color=(0, 0, 255), thickness=2)
                    # img.draw_string_advanced(filtered_center[0] + 10, filtered_center[1] - 10, 16, "C", color=(255, 255, 255, 0))

                    # 打印坐标信息（包含原始和滤波后的坐标）
                    if max_rect:
                        # print("=== 最大矩形框信息 ===")
                        # print(f"矩形尺寸: {max_rect.w()} x {max_rect.h()}")
                        # print(f"矩形面积: {calculate_area(max_rect)}")
                        # print(f"矩形边界: x={max_rect.x()}, y={max_rect.y()}, w={max_rect.w()}, h={max_rect.h()}")
                        # print("原始四个角点坐标:")
                        # for i, corner in enumerate(corners):
                        #     print(f"  角点{i+1}: ({corner[0]}, {corner[1]})")
                        # print("滤波后四个角点坐标:")
                        # for i, corner in enumerate(filtered_corners):
                        #     print(f"  角点{i+1}: ({corner[0]}, {corner[1]})")
                        # print(f"原始中心点坐标: ({center[0]}, {center[1]})")
                        # print(f"滤波后中心点坐标: ({filtered_center[0]}, {filtered_center[1]})")
                        # print(f"x误差: {x_error}, PID输出: {pid_output:.4f}")
                        # print(f"FPS: {fps.fps():.2f}")
                        # print("-" * 30)
                        pass
                else:
                    # 如果滤波器还没有足够的数据，在原图上使用原始坐标
                    if max_rect:
                        # 使用字典格式的矩形数据绘制矩形框
                        x, y, w, h = max_rect['x'], max_rect['y'], max_rect['w'], max_rect['h']
                        img.draw_rectangle([x, y, w, h], color=(0, 255, 0), thickness=2)
                        for i, corner in enumerate(corners):
                            img.draw_circle(corner[0], corner[1], 5, color=(255, 0, 0), thickness=2)
                            # img.draw_string_advanced(corner[0] + 8, corner[1] - 8, 16, str(i+1), color=(255, 255, 255, 0))
                        if center:
                            img.draw_circle(center[0], center[1], 8, color=(0, 0, 255), thickness=2)
                            # img.draw_string_advanced(center[0] + 10, center[1] - 10, 16, "C", color=(255, 255, 255, 0))
            else:
                # 没有找到符合条件的矩形，重置滤波器
                coord_filter.reset()
                # print(f"未检测到矩形 - FPS: {fps.fps():.2f}")

            # 在原图上绘制图像中心点
            img.draw_circle(IMAGE_CENTER_X, DETECT_HEIGHT // 2, 3, color=(255, 255, 0), thickness=2, fill=False)
            # img.draw_string(IMAGE_CENTER_X + 5, DETECT_HEIGHT // 2 - 10, "IMG_CENTER", color=(255, 255, 0), scale=1)

            # 显示FPS和显示模式信息 Display FPS and display mode info
            fps_text = f"FPS: {fps.fps():.1f}"
            mode_text = f"Display: {DISPLAY_MODE}"
            img.draw_string_advanced(5, 5, 16, fps_text, color=(255, 255, 255, 0))
            img.draw_string_advanced(5, 25, 16, mode_text, color=(255, 255, 255, 0))

            # 显示矩形检测状态信息
            # if rects:
            #     qualified_count = len(filtered_rects) if 'filtered_rects' in locals() else 0
            #     total_count = len(rects)
            #     status_text = f"Rects: {qualified_count}/{total_count}"
            #     # img.draw_string_advanced(5, 45, 16, status_text, color=(0, 255, 0, 0))

            #     if qualified_count > 0:
            #         img.draw_string_advanced(5, 65, 16, "Qualified!", color=(0, 255, 0, 0))
            #     else:
            #         img.draw_string_advanced(5, 65, 16, "No Qualified", color=(255, 0, 0, 0))
            # else:
            #     img.draw_string_advanced(5, 45, 16, "No Rects", color=(255, 0, 0, 0))

            # 显示误差值 Display error value
            if 'x_error' in locals():
                error_text = f"Error: {x_error:.1f}"
                # 在屏幕右上角显示误差信息
                img.draw_string_advanced(DETECT_WIDTH - 150, 5, 16, error_text, color=(0, 255, 255, 0))

            # 将原图结果显示到屏幕 Display the original image to screen
            if DISPLAY_MODE == "LCD":
                # LCD模式下居中显示图像 Center display image in LCD mode
                x = int((DISPLAY_WIDTH - DETECT_WIDTH) / 2)
                y = int((DISPLAY_HEIGHT - DETECT_HEIGHT) / 2)
                Display.show_image(img, x=x, y=y)
            else:
                # 虚拟显示模式 Virtual display mode
                Display.show_image(img)
            img = None

            gc.collect()

        except KeyboardInterrupt as e:
            # print("用户停止:", e)
            break
        except BaseException as e:
            # print(f"异常: {e}")
            break

def main():
    """主函数"""
    os.exitpoint(os.EXITPOINT_ENABLE)
    camera_is_init = False
    uart_is_init = False

    try:
        uart_init()
        uart_is_init = True

        camera_init()
        camera_is_init = True

        capture_picture()

    except Exception as e:
        # print(f"程序异常: {e}")
        pass
    finally:
        if camera_is_init:
            # print("正在释放摄像头资源...")
            camera_deinit()
        if uart_is_init:
            # print("正在释放UART串口资源...")
            uart_deinit()



if __name__ == "__main__":
    main()
