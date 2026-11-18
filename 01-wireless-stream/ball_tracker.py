"""
实验名称：钢球检测 + 单目测距 + UART输出（基于YOLO11）
实验平台：01Studio CanMV K230 / CanMV K230 mini
说明：
  1. 在 demo-camera.py 基础上，解析每个钢球的屏幕像素位置（左上角x,y + 宽w高h，
     800x480屏幕坐标系），并用相似三角形估计球到小车的直线距离。
  2. 三路输出：LCD屏叠加显示 / UART2(引脚11,12 @115200)发小车主控 / REPL串口打印。
  3. UART协议（带校验ASCII帧，详见 UART协议说明.md）：
       $B,n,x,y,w,h,dx,dy,dist_mm,flags[,...]*CS\r\n
     n=目标数(0~5)，无球时发空帧 $B,0*9E。
  4. 测距前必须先标定：运行 calibrate.py，把测得的 K 值填入下方 K_DIST。
"""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from libs.Utils import *
from media.sensor import *
from machine import UART, FPIOA
import gc
import time

# ============================ 配置区 ============================
# --- 模型/显示（与 demo-camera.py 保持一致）---
kmodel_path = "/sdcard/yolo11n_det_320.kmodel"
labels = {0: 'SteelBall'}
model_input_size = [320, 320]
display = "lcd2_4"                 # "hdmi" / "lcd3_5" / "lcd2_4"
rgb888p_size = [640, 480]
confidence_threshold = 0.45          # 小目标适当降低；ROI 会抑制管外误检
nms_threshold = 0.45

# --- 白色长管检测区域（归一化坐标，适配不同显示分辨率）---
# 根据当前三张实拍图设定：横向排除两端堵头，纵向包住整根白管。
ROI_X0 = 0.04
ROI_Y0 = 0.41
ROI_X1 = 0.96
ROI_Y1 = 0.68

# --- 测距 ---
K_DIST = 587.50                       # 标定常数(cm·px)：dist_cm = K_DIST / ((w+h)/2)
                                   # 先跑 calibrate.py 测得后填入；0 = 未标定(距离全无效)
DIST_MIN_CM = 5.0                  # 距离有效区间下限
DIST_MAX_CM = 150.0                # 距离有效区间上限(按真机实测稳定检出距离调整)

# --- 多目标跟踪 / 滑动平均 ---
SMOOTH_WIN = 5                     # 滑动平均窗口(帧)
LOST_FRAMES = 5                    # 连续丢失N帧后删除轨迹
MATCH_GATE = 80                    # 球心匹配门限(px，曼哈顿距离)
N_MAX = 5                          # 输出目标上限
MAX_DETS_IN = 8                    # 进入匹配前的检测数上限(面积降序预筛)
MAX_TRACKS = 8                     # 轨迹数硬顶
MIN_HITS = 2                       # 新轨迹需连续命中N帧才输出(防交叉/误检产生幻影目标)
EDGE_MARGIN = 2                    # 边缘截断判定边距(px)

# --- UART ---
UART_TX_PIN = 11                   # UART2: 11=TX, 12=RX
UART_RX_PIN = 12
UART_BAUD = 115200
UART_SEND_DIV = 1                  # 每N帧发一次(1=每帧都发)
REPL_PRINT = True                  # REPL回显开关(关掉可省打印耗时)

# --- 显示颜色 (R,G,B) ---
CROSS_COLOR = (255, 255, 0)        # 准星
OK_COLOR = (0, 255, 0)             # 有效目标
BAD_COLOR = (255, 0, 0)            # 无效目标/警告
TEXT_COLOR = (255, 255, 255)       # 状态栏
ROI_COLOR = (0, 255, 255)          # 检测区域边框
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

# 屏幕参数（初始化后由 pl.get_display_size() 刷新，供 calc_flags/绘制使用）
DISP_W = display_size[0]
DISP_H = display_size[1]
CENTER_X = DISP_W // 2
CENTER_Y = DISP_H // 2

# ------------------------- 帧工具 -------------------------

def scaled_roi(width, height):
    """将归一化 ROI 转为当前图像/显示坐标。"""
    x = int(width * ROI_X0)
    y = int(height * ROI_Y0)
    x2 = int(width * ROI_X1)
    y2 = int(height * ROI_Y1)
    return x, y, max(1, x2 - x), max(1, y2 - y)

def mask_outside_roi(img):
    """推理前将 RGBP888 ndarray 的白管以外区域置黑。"""
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
    """仅保留中心落在白管内的检测框。"""
    rx, ry, rw, rh = scaled_roi(DISP_W, DISP_H)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return rx <= cx < rx + rw and ry <= cy < ry + rh

def draw_detection_roi(osd):
    x, y, w, h = scaled_roi(DISP_W, DISP_H)
    osd.draw_rectangle(x, y, w, h, color=ROI_COLOR, thickness=2)

def checksum(body):
    """'$'与'*'之间所有字符逐字节累加 mod 256，返回两位大写hex"""
    s = 0
    for ch in body:
        s = (s + ord(ch)) & 0xFF
    return "%02X" % s

def pack_frame(targets):
    """targets: [(x,y,w,h,dx,dy,dist_mm,flags), ...] → 完整ASCII帧"""
    body = "B,%d" % len(targets)
    for t in targets:
        body += ",%d,%d,%d,%d,%d,%d,%d,%d" % t
    return "$%s*%s\r\n" % (body, checksum(body))

# ------------------------- 多目标跟踪器 -------------------------

tracks = []                        # 全局轨迹列表
next_track_id = 1

def new_track():
    global next_track_id
    t = {'id': next_track_id,
         'cx': 0.0, 'cy': 0.0, 'w': 0.0, 'h': 0.0, 'dist': 0.0,
         'win': [], 'lost': 0, 'hits': 0}
    next_track_id += 1
    return t

def push_sample(t, cx, cy, w, h, dist):
    """样本入滑动窗并重算均值（5帧滑动平均滤波）"""
    win = t['win']
    win.append((cx, cy, w, h, dist))
    if len(win) > SMOOTH_WIN:
        win.pop(0)
    n = len(win)
    sx = sy = sw = sh = sd = 0.0
    for s in win:
        sx += s[0]
        sy += s[1]
        sw += s[2]
        sh += s[3]
        sd += s[4]
    t['cx'] = sx / n
    t['cy'] = sy / n
    t['w'] = sw / n
    t['h'] = sh / n
    t['dist'] = sd / n
    t['lost'] = 0
    t['hits'] += 1

def tracker_update(dets):
    """dets: [(cx,cy,w,h,dist,area),...] 已面积降序 → 返回本帧可输出的track列表"""
    matched_ids = []
    out = []
    for d in dets:
        # 贪心最近邻匹配（球心曼哈顿距离，省sqrt）
        best = None
        best_d = MATCH_GATE
        for t in tracks:
            if t['id'] in matched_ids:
                continue
            dd = abs(t['cx'] - d[0]) + abs(t['cy'] - d[1])
            if dd < best_d:
                best = t
                best_d = dd
        if best is None:
            if len(tracks) >= MAX_TRACKS:
                continue           # 轨迹数硬顶：拒绝新建，防爆炸
            best = new_track()
            tracks.append(best)
        push_sample(best, d[0], d[1], d[2], d[3], d[4])
        matched_ids.append(best['id'])
        if best['hits'] >= MIN_HITS:
            out.append(best)
    # 未认领轨迹 lost+1，超时删除（保留5帧以便短暂遮挡后原id续上）
    i = 0
    while i < len(tracks):
        if tracks[i]['id'] not in matched_ids:
            tracks[i]['lost'] += 1
        if tracks[i]['lost'] > LOST_FRAMES:
            tracks.pop(i)
        else:
            i += 1
    # 输出按平滑后面积降序（插入排序，元素少且固件兼容性好）
    for i in range(1, len(out)):
        cur = out[i]
        ca = cur['w'] * cur['h']
        j = i - 1
        while j >= 0 and out[j]['w'] * out[j]['h'] < ca:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = cur
    return out[:N_MAX]

# ------------------------- 判定 -------------------------

def calc_flags(x, y, w, h, dist_cm):
    """→ (flags, dist_mm)。flags: bit0=valid, bit1=edge(贴边), bit2=range(距离超程)"""
    if dist_cm <= 0:
        dist_mm = 0
    else:
        dist_mm = int(dist_cm * 10 + 0.5)
        if dist_mm > 9999:
            dist_mm = 9999
    rx, ry, rw, rh = scaled_roi(DISP_W, DISP_H)
    edge = (x <= rx + EDGE_MARGIN or y <= ry + EDGE_MARGIN or
            x + w >= rx + rw - EDGE_MARGIN or
            y + h >= ry + rh - EDGE_MARGIN)
    in_range = (K_DIST > 0 and DIST_MIN_CM <= dist_cm <= DIST_MAX_CM)
    valid = (w > 0 and h > 0 and in_range and not edge)
    flags = (1 if valid else 0) | (2 if edge else 0) | (0 if in_range else 4)
    return flags, dist_mm

# ------------------------- LCD 叠加绘制 -------------------------

def draw_crosshair(osd):
    """过画面中心的全屏十字准星 + 中心小圆（黄色细线）"""
    osd.draw_line(0, CENTER_Y, DISP_W - 1, CENTER_Y, color=CROSS_COLOR, thickness=1)
    osd.draw_line(CENTER_X, 0, CENTER_X, DISP_H - 1, color=CROSS_COLOR, thickness=1)
    osd.draw_circle(CENTER_X, CENTER_Y, 6, color=CROSS_COLOR, thickness=1)

def draw_target(osd, x, y, w, h, cx_i, cy_i, flags, dist_mm):
    """球心十字 + 框下方距离文本（有效绿/无效红）"""
    color = OK_COLOR if (flags & 1) else BAD_COLOR
    osd.draw_cross(cx_i, cy_i, color=color, size=16, thickness=2)
    ty = y + h + 5
    if ty > DISP_H - 28:
        ty = DISP_H - 28           # 防文本出屏
    if flags & 1:
        txt = "D:%dcm" % (dist_mm // 10)
    else:
        txt = "D:---"
    osd.draw_string_advanced(x, ty, 24, txt, color=color)

# ------------------------- 初始化 -------------------------

pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
if display == "lcd2_4":
    pl.create(sensor=Sensor(id=2, width=1280, height=960))
else:
    pl.create(sensor=Sensor(id=2, width=1920, height=1080))

display_size = pl.get_display_size()
DISP_W = display_size[0]
DISP_H = display_size[1]
CENTER_X = DISP_W // 2
CENTER_Y = DISP_H // 2

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

fpioa = FPIOA()
fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)
uart = UART(UART.UART2, UART_BAUD)

clock = time.clock()
frame_cnt = 0

# ------------------------- 主循环 -------------------------

try:
    while True:
        clock.tick()
        img = pl.get_frame()
        mask_outside_roi(img)
        res = yolo.run(img)
        # 不调用 draw_result，避免把 ROI 外的原始 YOLO 框画到屏幕上。
        pl.osd_img.clear()

        # 1) 解析检测：检测框坐标位于当前显示坐标系。
        dets = []
        if res and len(res[0]) > 0:
            for i in range(len(res[0])):
                x_f, y_f, w_f, h_f = res[0][i]
                if not box_center_in_roi(x_f, y_f, w_f, h_f):
                    continue
                cx = x_f + w_f / 2.0
                cy = y_f + h_f / 2.0
                s = (w_f + h_f) / 2.0
                if K_DIST > 0 and s > 0:
                    dist = K_DIST / s        # 相似三角形单目测距
                else:
                    dist = 0.0
                dets.append((cx, cy, w_f, h_f, dist, w_f * h_f))
        # 面积降序（插入排序）+ 截断
        for i in range(1, len(dets)):
            cur = dets[i]
            j = i - 1
            while j >= 0 and dets[j][5] < cur[5]:
                dets[j + 1] = dets[j]
                j -= 1
            dets[j + 1] = cur
        if len(dets) > MAX_DETS_IN:
            dets = dets[:MAX_DETS_IN]

        # 2) 跟踪 + 滑动平均
        out = tracker_update(dets)

        # 3) 逐目标计算输出量 + flags
        targets = []
        for t in out:
            cx_i = int(round(t['cx']))
            cy_i = int(round(t['cy']))
            w_i = int(round(t['w']))
            h_i = int(round(t['h']))
            x_i = cx_i - w_i // 2
            y_i = cy_i - h_i // 2
            dx = cx_i - CENTER_X
            dy = cy_i - CENTER_Y
            flags, dist_mm = calc_flags(x_i, y_i, w_i, h_i, t['dist'])
            targets.append((x_i, y_i, w_i, h_i, dx, dy, dist_mm, flags))

        # 4) LCD 自定义叠加（全部在 draw_result 之后）
        draw_detection_roi(pl.osd_img)
        draw_crosshair(pl.osd_img)
        for tg in targets:
            draw_target(pl.osd_img, tg[0], tg[1], tg[2], tg[3],
                        tg[0] + tg[2] // 2, tg[1] + tg[3] // 2, tg[7], tg[6])
        pl.osd_img.draw_string_advanced(
            4, 4, 24, "FPS:%d N:%d" % (int(clock.fps()), len(targets)), color=TEXT_COLOR)
        if K_DIST <= 0:
            pl.osd_img.draw_string_advanced(4, 34, 32, "K NOT CALIBRATED!", color=BAD_COLOR)

        # 5) 组帧 + UART/REPL 双路输出
        frame = pack_frame(targets)
        if len(frame) > 200:                 # 帧长防线：异常超长降级为空帧
            frame = "$B,0*9E\r\n"
        frame_cnt += 1
        if frame_cnt % UART_SEND_DIV == 0:
            try:
                uart.write(frame)
            except Exception:
                pass                         # 发送异常不影响视觉主循环
        if REPL_PRINT:
            print(frame[:-2])                # 与UART内容一致(去掉\r\n)，便于对照

        pl.show_image()
        gc.collect()

except KeyboardInterrupt:
    pass
finally:
    # 释放资源
    yolo.deinit()
    pl.destroy()
