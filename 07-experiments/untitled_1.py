import time, os, sys
from media.sensor import *
from media.display import *
from media.media import *
from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart

# 初始化串口、协议等
uart = YbUart(baudrate=115200)
pto = YbProtocol()

# 模式保持为3，但不再通过按键切换（或根据需要保留切换逻辑）
mode = 3

# 显示参数
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# 初始化传感器
sensor = Sensor(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, fps=30)
sensor.reset()
time.sleep_ms(100)
sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
sensor.set_pixformat(sensor.RGB565)

# 初始化显示
Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
MediaManager.init()
sensor.run()

clock = time.clock() # 用于统计帧率

# --- 以下变量不再作为状态锁，仅用于存储临时数据 ---
# 如果需要在丢失信号时保持上一帧数据，可以在此定义全局缓存变量
# last_corners = None

def draw_first_rect(img, corners, shrink_ratio_x, shrink_ratio_y):
    # 代码逻辑保持不变（坐标规整与缩放）
    new_corners = [list(pt) for pt in corners]
    changed = True
    # ... (中间的坐标平滑合并逻辑保持不变，为节省篇幅折叠) ...
    # [此处保留原代码中 draw_first_rect 函数的所有内部逻辑]
    # ...

    # 计算中心点
    cx = sum([pt[0] for pt in corners_sorted]) / 4
    cy = sum([pt[1] for pt in corners_sorted]) / 4

    # 分别按x和y方向缩小
    shrinked = []
    for x, y in corners_sorted:
        new_x = int(cx + (x - cx) * shrink_ratio_x)
        new_y = int(cy + (y - cy) * shrink_ratio_y)
        shrinked.append([new_x, new_y])

    # 画小矩形
    for i in range(4):
        x1, y1 = shrinked[i]
        x2, y2 = shrinked[(i+1)%4]
        img.draw_line(x1, y1, x2, y2, color=(255, 255, 255)) # 白线
        img.draw_circle(x1, y1, 3, color=(0, 255, 0)) # 绿点

    return shrinked

# --- 主循环 ---
while True:
    clock.tick() # 开始计时

    # 1. 图像采集
    img = sensor.snapshot()
    img = img.gaussian(1) # 高斯滤波去噪

    # 2. 矩形检测 (Mode 3 Logic - 实时版)
    if mode == 3:
        # --- 核心改动：移除状态锁，直接执行检测 ---
        found_rect = False
        for rect in img.find_rects(threshold=20000, x_gradient=10, y_gradient=10):
            area = rect.magnitude()

            # 面积筛选 (保留原逻辑)
            if area > 80000:
                # 1. 获取角点
                corners = rect.corners()
                print(f"Detected Area: {area}")

                # 2. 绘制外框 (实时绘制)
                # 绘制四条边
                img.draw_line(corners[0][0], corners[0][1], corners[1][0], corners[1][1], color=(255, 255, 255))
                img.draw_line(corners[1][0], corners[1][1], corners[2][0], corners[2][1], color=(255, 255, 255))
                img.draw_line(corners[2][0], corners[2][1], corners[3][0], corners[3][1], color=(255, 255, 255))
                img.draw_line(corners[3][0], corners[3][1], corners[0][0], corners[0][1], color=(255, 255, 255))

                # 绘制顶点
                for p in corners:
                    img.draw_circle(p[0], p[1], 3, color=(255, 0, 0))

                # 3. 计算并绘制内框
                inner_corners = draw_first_rect(img, corners, shrink_ratio_x=0.86, shrink_ratio_y=0.87)

                # 4. (可选) 数据发送
                # 如果需要每一帧都发数据，取消下面的注释
                # process_and_pack_corners(corners, inner_corners)

                found_rect = True
                break # 找到一个符合条件的矩形后跳出循环，处理下一个帧

        # --- 可选逻辑：处理未检测到矩形的情况 ---
        # 如果你想在没识别到矩形时显示提示，或者发送特定数据包，可以在这里添加
        if not found_rect:
            # print("No rect detected this frame")
            # img.draw_string(10, 10, "Lost Target", color=(255, 0, 0))
            pass

    # 3. 显示与帧率控制
    Display.show_image(img)

    # 打印当前帧率 (调试用)
    # print(f"FPS: {clock.fps()}")

    # 注意：不要在这里加大的 sleep，否则会降低识别 FPS
    # time.sleep_ms(1) 保持或删除均可，主要靠 sensor.fps(30) 来控制
