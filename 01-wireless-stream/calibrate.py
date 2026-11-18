"""
实验名称：钢球测距标定工具（配合 ball_tracker.py 使用）
实验平台：01Studio CanMV K230 / CanMV K230 mini

标定原理：
    相似三角形：dist_cm = K / s，其中 s = (w+h)/2 为当前显示坐标系的检测框像素尺寸。
    把球放在镜头正前方已知距离 CAL_DIST_CM 处，测得稳定的 s0，则 K = CAL_DIST_CM * s0。
    注意：640x360→800x480 是非均匀拉伸，但两轴系数为常数，标定会自动吸收——
    前提是本脚本的 s 定义、PipeLine/显示参数与 ball_tracker.py 完全一致（勿改）。

操作步骤：
    1. 修改下方 CAL_DIST_CM 为实际摆放距离（推荐 50cm，量镜头到球心的直线距离）。
    2. 运行本脚本，把钢球固定在镜头正前方该距离处，尽量居于画面中心。
    3. 观察屏幕/串口的 K 值，等其稳定 1~2 秒后抄下来。
    4. 把 K 填入 ball_tracker.py 配置区的 K_DIST。
    5. 再把球放到 30cm / 100cm 处跑 ball_tracker.py 复测距离误差。
"""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from libs.Utils import *
from media.sensor import *
import gc
import time

# ============================ 配置区 ============================
CAL_DIST_CM = 50.0                 # 标定时球到镜头的真实距离(cm)
AVG_FRAMES = 30                    # 显示平均窗口(帧)

# --- 模型/显示（必须与 ball_tracker.py 完全一致）---
kmodel_path = "/sdcard/yolo11n_det_320.kmodel"
labels = {0: 'SteelBall'}
model_input_size = [320, 320]
display = "lcd2_4"                 # "hdmi" / "lcd3_5" / "lcd2_4"
rgb888p_size = [640, 480]
confidence_threshold = 0.45          # 与 ball_tracker.py 一致
nms_threshold = 0.45

# 必须与 ball_tracker.py 保持一致。
ROI_X0 = 0.04
ROI_Y0 = 0.41
ROI_X1 = 0.96
ROI_Y1 = 0.68
ROI_COLOR = (0, 255, 255)
# ===============================================================

if display == "hdmi":
    display_mode = "hdmi"
    display_size = [1920, 1080]
elif display == "lcd3_5":
    display_mode = "st7701"
    display_size = [800, 480]
elif display == "lcd2_4":
    display_mode = "st7701"
    display_size = [640, 480]

pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
if display == "lcd2_4":
    pl.create(sensor=Sensor(id=2, width=1280, height=960))
else:
    pl.create(sensor=Sensor(id=2, width=1920, height=1080))

display_size = pl.get_display_size()
DISP_W = display_size[0]
DISP_H = display_size[1]

def scaled_roi(width, height):
    x = int(width * ROI_X0)
    y = int(height * ROI_Y0)
    x2 = int(width * ROI_X1)
    y2 = int(height * ROI_Y1)
    return x, y, max(1, x2 - x), max(1, y2 - y)

def mask_outside_roi(img):
    shape = img.shape
    ndim = len(shape)
    if ndim == 4:                    # (1, 3, H, W)
        ih = shape[2]
        iw = shape[3]
    elif ndim == 3:                  # (3, H, W)
        ih = shape[1]
        iw = shape[2]
    else:
        raise ValueError("unexpected RGBP888 shape: %s" % (shape,))

    x, y, w, h = scaled_roi(iw, ih)
    bottom = y + h
    right = x + w

    if ndim == 4:
        if y > 0:
            img[:, :, 0:y, :] = 0
        if bottom < ih:
            img[:, :, bottom:ih, :] = 0
        if x > 0:
            img[:, :, y:bottom, 0:x] = 0
        if right < iw:
            img[:, :, y:bottom, right:iw] = 0
    else:
        if y > 0:
            img[:, 0:y, :] = 0
        if bottom < ih:
            img[:, bottom:ih, :] = 0
        if x > 0:
            img[:, y:bottom, 0:x] = 0
        if right < iw:
            img[:, y:bottom, right:iw] = 0

def box_center_in_roi(x, y, w, h):
    rx, ry, rw, rh = scaled_roi(DISP_W, DISP_H)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return rx <= cx < rx + rw and ry <= cy < ry + rh

def draw_detection_roi(osd):
    x, y, w, h = scaled_roi(DISP_W, DISP_H)
    osd.draw_rectangle(x, y, w, h, color=ROI_COLOR, thickness=2)

yolo = YOLO11(
    task_type="detect",
    mode="video",
    kmodel_path=kmodel_path,
    labels=labels,
    rgb888p_size=rgb888p_size,
    model_input_size=model_input_size,
    display_size=display_size,
    conf_thresh=confidence_threshold,
    nms_thresh=nms_threshold,
    max_boxes_num=50,
    debug_mode=0,
)
yolo.config_preprocess()

clock = time.clock()
swin = []                          # s 的滑动平均窗口

print("calibrate start, CAL_DIST_CM=%.1f" % CAL_DIST_CM)

try:
    while True:
        clock.tick()
        img = pl.get_frame()
        mask_outside_roi(img)
        res = yolo.run(img)
        pl.osd_img.clear()
        draw_detection_roi(pl.osd_img)

        if res and len(res[0]) > 0:
            # 取面积最大的框
            best = None
            best_area = 0.0
            for i in range(len(res[0])):
                x_f, y_f, w_f, h_f = res[0][i]
                if not box_center_in_roi(x_f, y_f, w_f, h_f):
                    continue
                area = w_f * h_f
                if area > best_area:
                    best_area = area
                    best = (x_f, y_f, w_f, h_f)
            if best is None:
                pl.osd_img.draw_string_advanced(
                    4, 4, 32, "NO BALL", color=(255, 0, 0))
                print("no ball in ROI")
                pl.show_image()
                gc.collect()
                continue

            x_f, y_f, w_f, h_f = best
            s = (w_f + h_f) / 2.0           # s 定义必须与 ball_tracker.py 一致

            swin.append(s)
            if len(swin) > AVG_FRAMES:
                swin.pop(0)
            s_avg = 0.0
            for v in swin:
                s_avg += v
            s_avg /= len(swin)
            K = CAL_DIST_CM * s_avg

            pl.osd_img.draw_string_advanced(4, 4, 32, "s=%.1f px" % s_avg, color=(255, 255, 255))
            pl.osd_img.draw_string_advanced(4, 44, 40, "K=%.1f" % K, color=(255, 255, 0))
            pl.osd_img.draw_string_advanced(4, 96, 24, "cal:%.1fcm" % CAL_DIST_CM, color=(255, 255, 255))
            pl.osd_img.draw_rectangle(int(x_f), int(y_f), int(w_f), int(h_f),
                                      color=(0, 255, 0), thickness=2)
            print("s=%.2f K=%.2f" % (s_avg, K))
        else:
            pl.osd_img.draw_string_advanced(4, 4, 32, "NO BALL", color=(255, 0, 0))
            print("no ball")

        pl.show_image()
        gc.collect()

except KeyboardInterrupt:
    pass
finally:
    yolo.deinit()
    pl.destroy()
