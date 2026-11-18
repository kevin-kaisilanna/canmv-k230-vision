"""
实验名称：钢球检测 + 单目测距 + UART输出 + AP网页图传（基于YOLO11）
实验平台：01Studio CanMV K230 / CanMV K230 mini
说明：
  1. 在 demo-camera.py 基础上，解析每个钢球的屏幕像素位置（左上角x,y + 宽w高h，
     800x480屏幕坐标系），并用相似三角形估计球到小车的直线距离。
  2. 四路输出：LCD叠加 / UART2(引脚11,12 @115200) / REPL打印 / AP网页图传。
  3. UART协议（带校验ASCII帧，详见 UART协议说明.md）：
       $B,n,x,y,w,h,dx,dy,dist_mm,flags[,...]*CS\r\n
     n=目标数(0~5)，无球时发空帧 $B,0*9E。
  4. 测距前必须先标定：运行 calibrate.py，把测得的 K 值填入下方 K_DIST。
  5. AP图传使用独立摄像头通道和独立线程；识别主循环不做JPEG压缩、不等待网络。
     网页图传默认320x240@5fps，以保护钢球识别帧率。
"""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from libs.Utils import *
from media.sensor import *
from machine import UART, FPIOA
import gc
import time
import os
import socket
import network
import _thread
import uctypes

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


# --- AP 网页图传（独立线程，不阻塞识别主循环）---
AP_SSID = "K230-CAM"
AP_PASSWORD = "YOUR_AP_PASSWORD"           # Yahboom AP固件要求至少8个字符
HTTP_PORT = 8081
STREAM_SIZE = [320, 240]            # 低分辨率独立通道，优先保护识别FPS
STREAM_FPS = 5                      # 仅网页图传帧率，不改变AI识别帧率
STREAM_JPEG_QUALITY = 40
STREAM_DRAW_TARGETS = True          # 网页中叠加钢球框和中心点
STREAM_SOCKET_RETRY_MS = 5
STREAM_ACCEPT_POLL_MS = 100          # 无客户端时降低线程唤醒频率
STREAM_REQUEST_TIMEOUT_MS = 2000
STREAM_SEND_TIMEOUT_MS = 1500

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


# ------------------------- AP 网页图传 -------------------------

_CRLF = b"\r\n"

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>K230 钢球识别 AP 图传</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui,"Microsoft YaHei",sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #07111f; color: #e8f1ff; }
    main { width: min(920px,100%); margin: auto; padding: 16px; }
    h1 { margin: 4px 0; font-size: 24px; }
    .sub { color: #91a4bd; margin-bottom: 14px; }
    .viewer { position: relative; overflow: hidden; background: #000; border: 1px solid #263b55; border-radius: 14px; }
    #stream { display: block; width: 100%; min-height: 240px; object-fit: contain; }
    #state { position: absolute; left: 10px; top: 10px; padding: 5px 9px; border-radius: 99px; background: #17263dcc; }
    .panel { margin-top: 12px; padding: 14px; border: 1px solid #263b55; border-radius: 14px; background: #0d1b2d; }
    .buttons { display: flex; flex-wrap: wrap; gap: 9px; }
    button,a { border: 0; border-radius: 9px; padding: 10px 15px; font-size: 15px; color: white; background: #1976d2; text-decoration: none; cursor: pointer; }
    button:disabled,.disabled { opacity: .4; pointer-events: none; }
    #stop { background: #d43b4d; }
    #download { background: #198754; }
    .meta { margin-top: 11px; color: #a9bad0; line-height: 1.7; }
    #recordCanvas { display: none; }
  </style>
</head>
<body>
<main>
  <h1>K230 钢球识别 AP 图传</h1>
  <div class="sub">识别主循环不等待网络；网页图传由独立通道和独立线程处理</div>
  <div class="viewer">
    <img id="stream" src="/stream.mjpg" alt="钢球识别实时画面">
    <div id="state">正在连接…</div>
  </div>
  <div class="panel">
    <div class="buttons">
      <button id="start" disabled>开始录像</button>
      <button id="stop" disabled>停止录像</button>
      <a id="download" class="disabled">下载 MP4</a>
      <button id="snapshot" disabled>保存截图</button>
    </div>
    <div class="meta">录像格式：<span id="format">检测中…</span><br><span id="message">建议使用较新的 Chrome 或 Edge。</span></div>
  </div>
  <canvas id="recordCanvas"></canvas>
</main>
<script>
(() => {
  const img = document.getElementById('stream');
  const canvas = document.getElementById('recordCanvas');
  const ctx = canvas.getContext('2d', {alpha:false});
  const start = document.getElementById('start');
  const stop = document.getElementById('stop');
  const download = document.getElementById('download');
  const snapshot = document.getElementById('snapshot');
  const state = document.getElementById('state');
  const format = document.getElementById('format');
  const message = document.getElementById('message');
  let recorder = null, chunks = [], objectUrl = null, drawing = false;
  const candidates = [
    'video/mp4;codecs=avc1.42E01E',
    'video/mp4;codecs=avc1',
    'video/mp4;codecs=h264',
    'video/mp4'
  ];
  const hasRecorder = typeof window.MediaRecorder !== 'undefined';
  const mime = hasRecorder && typeof MediaRecorder.isTypeSupported === 'function'
    ? (candidates.find(t => MediaRecorder.isTypeSupported(t)) || '') : '';
  format.textContent = mime || '浏览器不支持 MP4';
  function draw() {
    if (img.naturalWidth) {
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    requestAnimationFrame(draw);
  }
  img.addEventListener('load', () => {
    if (!drawing) { drawing = true; requestAnimationFrame(draw); }
    state.textContent = '实时画面已连接';
    snapshot.disabled = false;
    const ok = hasRecorder && typeof canvas.captureStream === 'function' && Boolean(mime);
    start.disabled = !ok;
    message.textContent = ok ? 'MP4录像在电脑浏览器中完成，不写入K230存储卡。' : '浏览器不能直接编码MP4，请升级Chrome或Edge。';
  });
  img.addEventListener('error', () => {
    state.textContent = '图传断开，正在重连…'; start.disabled = true; snapshot.disabled = true;
    setTimeout(() => { img.src = '/stream.mjpg?t=' + Date.now(); }, 1200);
  });
  start.addEventListener('click', () => {
    if (!mime || typeof canvas.captureStream !== 'function') return;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    download.removeAttribute('href'); download.classList.add('disabled'); chunks = [];
    const mediaStream = canvas.captureStream(10);
    try {
      recorder = new MediaRecorder(mediaStream, {mimeType:mime, videoBitsPerSecond:2000000});
    } catch (e) {
      mediaStream.getTracks().forEach(t => t.stop());
      message.textContent = 'MP4录像器创建失败：' + (e.message || e); return;
    }
    recorder.addEventListener('dataavailable', e => { if (e.data && e.data.size) chunks.push(e.data); });
    recorder.addEventListener('stop', () => {
      const blob = new Blob(chunks, {type: recorder.mimeType || mime});
      objectUrl = URL.createObjectURL(blob);
      download.href = objectUrl;
      download.download = 'k230-steelball-' + new Date().toISOString().replace(/[:.]/g,'-') + '.mp4';
      download.classList.remove('disabled'); start.disabled = false; stop.disabled = true;
      state.textContent = '录像完成'; message.textContent = 'MP4已生成：' + (blob.size/1048576).toFixed(1) + ' MB';
      mediaStream.getTracks().forEach(t => t.stop());
    });
    recorder.start(1000); start.disabled = true; stop.disabled = false; state.textContent = '正在录像';
  });
  stop.addEventListener('click', () => { if (recorder && recorder.state !== 'inactive') recorder.stop(); stop.disabled = true; });
  snapshot.addEventListener('click', () => {
    ctx.drawImage(img,0,0,canvas.width,canvas.height);
    canvas.toBlob(blob => {
      if (!blob) return;
      const a=document.createElement('a'), u=URL.createObjectURL(blob);
      a.href=u; a.download='k230-steelball-'+new Date().toISOString().replace(/[:.]/g,'-')+'.jpg'; a.click();
      setTimeout(() => URL.revokeObjectURL(u),1000);
    },'image/jpeg',0.92);
  });
  window.addEventListener('beforeunload', () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });
})();
</script>
</body>
</html>
"""


def _error_number(error):
    number = getattr(error, "errno", None)
    if number is None:
        try:
            number = error.args[0]
        except Exception:
            number = None
    return number


def _would_block(error):
    return _error_number(error) in (11, 35, 10035)


def _get_ap_info(ap):
    try:
        return ap.info()
    except Exception:
        try:
            return ap.config("info")
        except Exception as error:
            return "unavailable: %s" % error


def create_ap(ssid, password):
    """严格沿用 Yahboom CanMV v1.8 的 ssid= + key= AP接口。"""
    if len(password) < 8:
        raise ValueError("AP password must contain at least 8 characters")
    ap = network.WLAN(network.AP_IF)
    try:
        ap.stop()
        time.sleep_ms(500)
    except Exception as error:
        print("[AP] previous hotspot stop skipped:", error)
    # 不要添加 password= 或 channel=，当前Yahboom固件会报 extra keyword arguments given。
    ap.config(ssid=ssid, key=password)
    time.sleep(3)
    print("[AP] SSID:", ssid)
    print("[AP] password:", password)
    print("[AP] info:", _get_ap_info(ap))
    print("[AP] ifconfig:", ap.ifconfig())
    return ap


def _send_all(sock, data):
    view = memoryview(data)
    sent_total = 0
    last_progress = time.ticks_ms()
    while sent_total < len(view):
        try:
            sent = sock.send(view[sent_total:])
        except OSError as error:
            if not _would_block(error):
                raise
            sent = None
        if sent is None:
            if time.ticks_diff(time.ticks_ms(), last_progress) >= STREAM_SEND_TIMEOUT_MS:
                raise OSError("socket send timed out")
            time.sleep_ms(STREAM_SOCKET_RETRY_MS)
            continue
        if sent <= 0:
            raise OSError("socket closed during send")
        sent_total += sent
        last_progress = time.ticks_ms()


def _send_response(client, status, content_type, body=b""):
    if isinstance(body, str):
        body = body.encode("utf-8")
    header = (
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Connection: close\r\n\r\n" % (status, content_type, len(body))
    ).encode("utf-8")
    _send_all(client, header)
    if body:
        _send_all(client, body)


def _read_request_path(client):
    request = bytearray()
    last_progress = time.ticks_ms()
    while len(request) < 4096:
        try:
            chunk = client.recv(512)
        except OSError as error:
            if not _would_block(error):
                raise
            chunk = None
        if chunk is None:
            if time.ticks_diff(time.ticks_ms(), last_progress) >= STREAM_REQUEST_TIMEOUT_MS:
                raise OSError("HTTP request timed out")
            time.sleep_ms(STREAM_SOCKET_RETRY_MS)
            continue
        if not chunk:
            break
        request.extend(chunk)
        last_progress = time.ticks_ms()
        if b"\r\n\r\n" in request:
            break
    if not request:
        return "/"
    parts = bytes(request).split(b"\r\n", 1)[0].split()
    if len(parts) < 2:
        return "/"
    try:
        return parts[1].decode("utf-8")
    except Exception:
        return "/"


class APStreamServer:
    """低优先级网页图传线程；识别循环从不等待该对象。"""

    def __init__(self, sensor, source_width, source_height):
        self.sensor = sensor
        self.source_width = source_width
        self.source_height = source_height
        self.latest_targets = []
        self.ap = None
        self.server = None
        self.client = None
        self.running = False
        self.thread_done = True

    def start(self):
        self.ap = create_ap(AP_SSID, AP_PASSWORD)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", HTTP_PORT))
        self.server.listen(3)
        self.server.setblocking(False)
        self.running = True
        self.thread_done = False
        _thread.start_new_thread(self._worker, ())
        ip_address = self.ap.ifconfig()[0]
        print("[READY] Connect Wi-Fi:", AP_SSID)
        print("[READY] Open browser: http://%s:%d/" % (ip_address, HTTP_PORT))
        print("[READY] AI loop is independent from the 320x240 MJPEG worker.")

    def stop(self):
        self.running = False
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass
            self.server = None
        deadline = time.ticks_add(time.ticks_ms(), STREAM_SEND_TIMEOUT_MS + 500)
        while not self.thread_done and time.ticks_diff(deadline, time.ticks_ms()) > 0:
            time.sleep_ms(20)
        # RT-Smart网络始终活动，不调用已弃用的 active(False)。
        self.ap = None

    def _draw_overlay(self, frame):
        if not STREAM_DRAW_TARGETS:
            return
        sx = STREAM_SIZE[0] / self.source_width
        sy = STREAM_SIZE[1] / self.source_height
        targets = self.latest_targets
        rx, ry, rw, rh = scaled_roi(STREAM_SIZE[0], STREAM_SIZE[1])
        frame.draw_rectangle(rx, ry, rw, rh, color=ROI_COLOR, thickness=1)
        cx0 = STREAM_SIZE[0] // 2
        cy0 = STREAM_SIZE[1] // 2
        frame.draw_line(cx0 - 6, cy0, cx0 + 6, cy0, color=CROSS_COLOR, thickness=1)
        frame.draw_line(cx0, cy0 - 6, cx0, cy0 + 6, color=CROSS_COLOR, thickness=1)
        for target in targets:
            x = int(target[0] * sx)
            y = int(target[1] * sy)
            w = max(1, int(target[2] * sx))
            h = max(1, int(target[3] * sy))
            flags = target[7]
            color = OK_COLOR if (flags & 1) else BAD_COLOR
            frame.draw_rectangle(x, y, w, h, color=color, thickness=2)
            cx = x + w // 2
            cy = y + h // 2
            frame.draw_line(cx - 4, cy, cx + 4, cy, color=color, thickness=1)
            frame.draw_line(cx, cy - 4, cx, cy + 4, color=color, thickness=1)

    def _stream_mjpeg(self, client):
        _send_all(client, (
            b"HTTP/1.1 200 OK\r\n"
            b"Cache-Control: no-cache, private\r\n"
            b"Pragma: no-cache\r\n"
            b"Connection: close\r\n"
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        ))
        interval_ms = max(1, 1000 // STREAM_FPS)
        print("[HTTP] MJPEG client connected")
        while self.running:
            started = time.ticks_ms()
            # CAM_CHN_ID_1 是专用低分辨率图传通道；AI仍由PipeLine自己的RGB888P通道处理。
            frame = self.sensor.snapshot(chn=CAM_CHN_ID_1)
            if frame == -1:
                time.sleep_ms(5)
                continue
            self._draw_overlay(frame)
            jpeg = frame.compressed(quality=STREAM_JPEG_QUALITY)
            jpeg_size = jpeg.size()
            jpeg_data = uctypes.bytearray_at(jpeg.virtaddr(), jpeg_size)
            part_header = (
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % jpeg_size
            ).encode("ascii")
            _send_all(client, part_header)
            _send_all(client, jpeg_data)
            _send_all(client, _CRLF)
            del jpeg_data
            del jpeg
            del frame
            remaining = interval_ms - time.ticks_diff(time.ticks_ms(), started)
            if remaining > 0:
                time.sleep_ms(remaining)

    def _serve_client(self, client, address):
        try:
            client.setblocking(False)
            path = _read_request_path(client).split("?", 1)[0]
            print("[HTTP]", address, path)
            if path == "/" or path == "/index.html":
                _send_response(client, "200 OK", "text/html; charset=utf-8", INDEX_HTML)
            elif path == "/stream.mjpg":
                self._stream_mjpeg(client)
            elif path == "/health":
                _send_response(client, "200 OK", "application/json", b'{"ok":true}')
            elif path == "/favicon.ico":
                _send_response(client, "204 No Content", "image/x-icon")
            else:
                _send_response(client, "404 Not Found", "text/plain; charset=utf-8", "404 Not Found")
        except OSError as error:
            if self.running:
                print("[HTTP] client disconnected:", error)
        except Exception as error:
            if self.running:
                print("[HTTP] client error:", error)
        finally:
            try:
                client.close()
            except Exception:
                pass
            self.client = None

    def _worker(self):
        try:
            while self.running:
                try:
                    client, address = self.server.accept()
                except OSError as error:
                    if not _would_block(error) and self.running:
                        print("[HTTP] accept error:", error)
                    time.sleep_ms(STREAM_ACCEPT_POLL_MS)
                    continue
                self.client = client
                self._serve_client(client, address)
        except Exception as error:
            if self.running:
                print("[HTTP] worker error:", error)
        finally:
            self.thread_done = True


# ------------------------- 初始化 -------------------------

pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
if display == "lcd2_4":
    pl.create(sensor=Sensor(id=2, width=1280, height=960), ch1_frame_size=STREAM_SIZE)
else:
    pl.create(sensor=Sensor(id=2, width=1920, height=1080), ch1_frame_size=STREAM_SIZE)

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

# AP失败时只关闭网页图传，钢球识别、LCD和UART仍继续运行。
stream_server = None
try:
    stream_server = APStreamServer(pl.sensor, DISP_W, DISP_H)
    stream_server.start()
except Exception as error:
    print("[AP] web stream disabled:", error)
    if stream_server is not None:
        try:
            stream_server.stop()
        except Exception:
            pass
    stream_server = None

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

        # 只替换一个已完整构建的列表引用；不复制图像、不压缩JPEG、不等待网络。
        if stream_server is not None:
            stream_server.latest_targets = targets

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
    # 先停止图传线程，再释放其共享的摄像头/PipeLine资源。
    if stream_server is not None:
        try:
            stream_server.stop()
        except Exception as error:
            print("[HTTP] stop error:", error)
    yolo.deinit()
    pl.destroy()
