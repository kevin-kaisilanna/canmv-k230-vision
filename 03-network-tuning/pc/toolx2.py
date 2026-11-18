import cv2
import numpy as np
from lib.h265 import VideoStreamHandler
from lib.receiver import DataReceiver
from lib.justfloat import JustFloatEncoder

# 尝试导入本地配置
try:
    import config
    SLIDER_COUNT = config.SLIDER_COUNT
    SLIDER_NAME_PREFIX = config.SLIDER_NAME_PREFIX
    SLIDER_INITIAL_VALUE = config.SLIDER_INITIAL_VALUE
    WINDOW_NAME = config.WINDOW_NAME
    WINDOW_WIDTH = config.WINDOW_WIDTH
    WINDOW_HEIGHT = config.WINDOW_HEIGHT
    BIN_BLOCK_SIZE = config.BIN_BLOCK_SIZE
    BIN_C = config.BIN_C
except ImportError:
    SLIDER_COUNT = 6
    SLIDER_NAME_PREFIX = "Channel"
    SLIDER_INITIAL_VALUE = 128
    WINDOW_NAME = "H265 Stream"
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 600
    BIN_BLOCK_SIZE = 11
    BIN_C = 2

# 创建视频处理器和接收器
video_handler = VideoStreamHandler()
receiver = DataReceiver(video_handler)

# 滑条状态
slider_values = [SLIDER_INITIAL_VALUE] * SLIDER_COUNT
slider_names = [f"{SLIDER_NAME_PREFIX} {i}" for i in range(SLIDER_COUNT)]

# 创建OpenCV窗口（支持缩放）
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

# 创建滑条
for i, name in enumerate(slider_names):
    cv2.createTrackbar(name, WINDOW_NAME, slider_values[i], 255,
                      lambda v, idx=i: slider_callback(idx, v))

def slider_callback(idx, value):
    """滑条回调函数"""
    global slider_values
    slider_values[idx] = value

def send_slider_values():
    """发送滑条值"""
    # 归一化到[0, 1]范围
    normalized = [v for v in slider_values]
    data = JustFloatEncoder.encode(normalized)
    result = receiver.send_to_device(data)
    
    status = {
        1: "Sent",
        0: "Queue full",
        -1: "No connection"
    }.get(result, "Unknown")
    print(f"Slider values sent: {slider_values} -> {status}")

def apply_binarization(frame):
    """应用二值化处理"""
    if frame is None:
        return None
        
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值二值化 - 使用高斯自适应
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        BIN_BLOCK_SIZE,  # 邻域大小
        BIN_C            # 常数减数
    )
    
    # 转换为BGR格式以便与原始图像并排显示
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# 启动接收器
receiver.start()

# 主循环
try:
    while True:
        # 获取视频帧
        frame = video_handler.get_frame()
        
        if frame is not None:
            # 应用二值化处理
            binary_frame = apply_binarization(frame)
            
            # 并排显示原始图像和二值化图像
            combined = np.hstack((frame, binary_frame))
            
            # 显示组合图像
            cv2.imshow(WINDOW_NAME, combined)
        
        # 处理按键事件
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):  # 按's'键发送滑条值
            send_slider_values()
        elif key == 27:  # ESC键退出
            break

finally:
    receiver.stop()
    cv2.destroyAllWindows()
    print("Application exited")
    