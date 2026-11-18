"""Grayscale steel-ball tracker with an independent AP web stream.

Detection remains on CAM_CHN_ID_0 at 640x67. A preallocated image buffer hands
an occasional frame to the HTTP thread, avoiding concurrent Sensor.snapshot()
calls across threads. The web page can view, capture JPG images, and record MP4.
"""

import gc
import time
import socket
import network
import _thread
import uctypes

import cv2
import image
import launcher_control
import ulab.numpy as np
from machine import FPIOA, UART
from media.display import *
from media.media import *
from media.sensor import *


CAMERA_ID = 2
CAMERA_FPS = 60

# Preserve the detector's original hardware crop and 640x67 grayscale input.
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
SENSOR_CROP_X = 0
SENSOR_CROP_Y = 338
SENSOR_CROP_W = 1280
SENSOR_CROP_H = 134

FRAME_WIDTH = 640
FRAME_HEIGHT = 67
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

REFERENCE_FRAME_WIDTH = 640
REFERENCE_FRAME_HEIGHT = 480
CAPTURE_ORIGIN_X = 0
CAPTURE_ORIGIN_Y = 229
CAMERA_CENTER_X = REFERENCE_FRAME_WIDTH // 2
CAMERA_CENTER_Y = REFERENCE_FRAME_HEIGHT // 2

BALL_DIAMETER_MM = 10.0
BALL_DIAMETER_PIXELS = 20.0

ROI_X = 0
ROI_Y = 0
ROI_W = FRAME_WIDTH
ROI_H = FRAME_HEIGHT

HOUGH_DP = 1.0
HOUGH_MIN_DISTANCE = 18
HOUGH_CANNY_THRESHOLD = 60
HOUGH_SCORE_THRESHOLD = 12
MIN_RADIUS = 4
MAX_RADIUS = 30
GAUSSIAN_KERNEL = 3
GAUSSIAN_SIGMA = 0.8
ENABLE_HOUGH_REFINEMENT = False
HOUGH_EVERY_N_FRAMES = 5

BLACKHAT_KERNEL_SIZE = 21
BLACKHAT_BLUR_KERNEL = 5
MIN_BLACKHAT_RESPONSE = 12.0
MIN_TRACKED_RESPONSE = 7.0
FALLBACK_RADIUS = 10
BLACKHAT_CENTER_Y_OFFSET = -5

SEARCH_X_MARGIN = 10
SEARCH_Y_MIN = 8
SEARCH_Y_MAX = 60
MAX_ANCHOR_DISTANCE = 22
TRACK_SEARCH_X = 100
TRACK_SEARCH_Y = 28

MIN_LOCAL_CONTRAST = 2.0
MAX_CIRCLES = 20
EDGE_MARGIN = 3

CONFIRM_FRAMES = 1
LOST_HOLD_FRAMES = 1
MAX_CENTER_JUMP = 160
POSITION_ALPHA = 0.70
FAST_POSITION_ALPHA = 0.92
FAST_MOTION_THRESHOLD = 6.0
RADIUS_ALPHA = 0.35
VELOCITY_ALPHA = 0.65
VELOCITY_DECAY = 0.75
DISPLAY_PREDICTION = 0.35
BOX_MARGIN = 3
AXIS_COLOR = (0, 0, 0)
DISPLAY_TO_IDE = True
DISPLAY_EVERY_N_FRAMES = 2
PRINT_EVERY_N_FRAMES = 60

# UART1: pin 9=TX, pin 10=RX.
UART_ENABLED = True
UART_TX_PIN = 9
UART_RX_PIN = 10
UART_BAUD = 115200
UART_SEND_EVERY_N_FRAMES = 1

# AP configuration must use ssid= and key= on Yahboom CanMV v1.8 firmware.
AP_SSID = "K230-CAM"
AP_PASSWORD = "YOUR_AP_PASSWORD"
HTTP_PORT = 8081

# The browser records to an even 640x240 canvas. K230 streams the detector's
# native narrow grayscale frame and does not perform an extra camera snapshot.
STREAM_WIDTH = FRAME_WIDTH
STREAM_HEIGHT = FRAME_HEIGHT
STREAM_FPS = 5
STREAM_FRAME_DIV = max(1, CAMERA_FPS // STREAM_FPS)
STREAM_JPEG_QUALITY = 42
STREAM_DRAW_RESULT = True
STREAM_ACCEPT_POLL_MS = 100
STREAM_SOCKET_RETRY_MS = 5
STREAM_REQUEST_TIMEOUT_MS = 2000
STREAM_SEND_TIMEOUT_MS = 1500


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>K230 灰度小球识别</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui,"Microsoft YaHei",sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #07111f; color: #e8f1ff; }
    main { width: min(920px,100%); margin: auto; padding: 16px; }
    h1 { margin: 4px 0; font-size: 24px; letter-spacing: 0; }
    .sub { color: #9fb0c6; margin: 4px 0 14px; }
    .viewer { position: relative; overflow: hidden; background: #000; border: 1px solid #29415e; border-radius: 8px; padding: 76px 0; }
    #stream { display: block; width: 100%; height: auto; image-rendering: auto; }
    #state { position: absolute; left: 10px; top: 10px; padding: 5px 9px; border-radius: 99px; background: #17263ddd; }
    .panel { margin-top: 12px; padding: 14px; border-top: 1px solid #29415e; background: #0d1b2d; }
    .buttons { display: flex; flex-wrap: wrap; gap: 9px; }
    button,a { border: 0; border-radius: 7px; padding: 10px 14px; font-size: 15px; color: #fff; background: #1976d2; text-decoration: none; cursor: pointer; }
    button:disabled,.disabled { opacity: .42; pointer-events: none; }
    #stop { background: #c93648; }
    #download { background: #17844f; }
    .meta { margin-top: 11px; color: #adbed2; line-height: 1.7; }
    #recordCanvas { display: none; }
    @media (max-width: 600px) { .viewer { padding: 42px 0; } button,a { flex: 1 1 calc(50% - 9px); text-align: center; } }
  </style>
</head>
<body>
<main>
  <h1>K230 灰度小球识别</h1>
  <div class="sub">AP 独立通道图传 · 识别主循环不等待网络</div>
  <div class="viewer">
    <img id="stream" src="/stream.mjpg" alt="灰度小球识别实时画面">
    <div id="state">正在连接…</div>
  </div>
  <section class="panel">
    <div class="buttons">
      <button id="start" disabled>开始录像</button>
      <button id="stop" disabled>停止录像</button>
      <a id="download" class="disabled">下载 MP4</a>
      <button id="snapshot" disabled>保存截图</button>
    </div>
    <div class="meta">录像格式：<span id="format">检测中…</span><br><span id="message">建议使用较新的 Chrome 或 Edge。</span></div>
  </section>
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
  let recorder = null, chunks = [], recordUrl = null, drawing = false;
  const candidates = [
    'video/mp4;codecs=avc1.42E01E',
    'video/mp4;codecs=avc1',
    'video/mp4;codecs=h264',
    'video/mp4'
  ];
  const hasRecorder = typeof window.MediaRecorder !== 'undefined';
  const mime = hasRecorder && typeof MediaRecorder.isTypeSupported === 'function'
    ? (candidates.find(type => MediaRecorder.isTypeSupported(type)) || '') : '';
  format.textContent = mime || '浏览器不支持 MP4';

  function drawFrame() {
    if (img.naturalWidth && img.naturalHeight) {
      const targetWidth = 640;
      const targetHeight = 240;
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }
      const scaledHeight = Math.round(img.naturalHeight * targetWidth / img.naturalWidth);
      const y = Math.floor((targetHeight - scaledHeight) / 2);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, targetWidth, targetHeight);
      ctx.drawImage(img, 0, y, targetWidth, scaledHeight);
    }
    requestAnimationFrame(drawFrame);
  }

  img.addEventListener('load', () => {
    if (!drawing) { drawing = true; requestAnimationFrame(drawFrame); }
    state.textContent = '实时画面已连接';
    snapshot.disabled = false;
    const canRecord = hasRecorder && typeof canvas.captureStream === 'function' && Boolean(mime);
    start.disabled = !canRecord;
    message.textContent = canRecord ? 'MP4在电脑浏览器中编码，不写入K230存储卡。' : '浏览器不能直接编码MP4，请升级Chrome或Edge。';
  });

  img.addEventListener('error', () => {
    if (recorder && recorder.state === 'recording') recorder.stop();
    state.textContent = '图传断开，正在重连…';
    start.disabled = true;
    snapshot.disabled = true;
    setTimeout(() => { img.src = '/stream.mjpg?t=' + Date.now(); }, 1200);
  });

  start.addEventListener('click', () => {
    if (!mime || typeof canvas.captureStream !== 'function') return;
    if (recordUrl) { URL.revokeObjectURL(recordUrl); recordUrl = null; }
    download.removeAttribute('href');
    download.classList.add('disabled');
    chunks = [];
    const mediaStream = canvas.captureStream(10);
    try {
      recorder = new MediaRecorder(mediaStream, {mimeType:mime, videoBitsPerSecond:1800000});
    } catch (error) {
      mediaStream.getTracks().forEach(track => track.stop());
      message.textContent = 'MP4录像器创建失败：' + (error.message || error);
      return;
    }
    recorder.addEventListener('dataavailable', event => {
      if (event.data && event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener('stop', () => {
      const blob = new Blob(chunks, {type:recorder.mimeType || mime});
      recordUrl = URL.createObjectURL(blob);
      download.href = recordUrl;
      download.download = 'k230-gray-ball-' + new Date().toISOString().replace(/[:.]/g,'-') + '.mp4';
      download.classList.remove('disabled');
      start.disabled = false;
      stop.disabled = true;
      state.textContent = '录像完成';
      message.textContent = 'MP4已生成：' + (blob.size / 1048576).toFixed(1) + ' MB';
      mediaStream.getTracks().forEach(track => track.stop());
    });
    recorder.start(1000);
    start.disabled = true;
    stop.disabled = false;
    state.textContent = '正在录像';
  });

  stop.addEventListener('click', () => {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    stop.disabled = true;
  });

  snapshot.addEventListener('click', () => {
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'k230-gray-ball-' + new Date().toISOString().replace(/[:.]/g,'-') + '.jpg';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, 'image/jpeg', 0.92);
  });

  window.addEventListener('beforeunload', () => {
    if (recordUrl) URL.revokeObjectURL(recordUrl);
  });
})();
</script>
</body>
</html>
"""


def validate_config():
    if BALL_DIAMETER_MM <= 0 or BALL_DIAMETER_PIXELS <= 0:
        raise ValueError("ball diameter calibration must be positive")
    if (SENSOR_CROP_X < 0 or SENSOR_CROP_Y < 0 or
            SENSOR_CROP_W <= 0 or SENSOR_CROP_H <= 0):
        raise ValueError("invalid sensor crop")
    if (SENSOR_CROP_X + SENSOR_CROP_W > SENSOR_WIDTH or
            SENSOR_CROP_Y + SENSOR_CROP_H > SENSOR_HEIGHT):
        raise ValueError("sensor crop exceeds source mode")
    if (CAPTURE_ORIGIN_X < 0 or CAPTURE_ORIGIN_Y < 0 or
            CAPTURE_ORIGIN_X + FRAME_WIDTH > REFERENCE_FRAME_WIDTH or
            CAPTURE_ORIGIN_Y + FRAME_HEIGHT > REFERENCE_FRAME_HEIGHT):
        raise ValueError("capture mapping exceeds reference frame")
    if ROI_X < 0 or ROI_Y < 0 or ROI_W <= 0 or ROI_H <= 0:
        raise ValueError("invalid ROI")
    if ROI_X + ROI_W > FRAME_WIDTH or ROI_Y + ROI_H > FRAME_HEIGHT:
        raise ValueError("ROI exceeds the camera frame")
    if GAUSSIAN_KERNEL < 3 or GAUSSIAN_KERNEL % 2 == 0:
        raise ValueError("GAUSSIAN_KERNEL must be an odd number >= 3")
    if BLACKHAT_KERNEL_SIZE < 3 or BLACKHAT_KERNEL_SIZE % 2 == 0:
        raise ValueError("BLACKHAT_KERNEL_SIZE must be odd and >= 3")
    if BLACKHAT_BLUR_KERNEL < 3 or BLACKHAT_BLUR_KERNEL % 2 == 0:
        raise ValueError("BLACKHAT_BLUR_KERNEL must be odd and >= 3")
    if (SEARCH_X_MARGIN < 0 or SEARCH_X_MARGIN * 2 >= ROI_W or
            SEARCH_Y_MIN < 0 or SEARCH_Y_MAX > ROI_H or
            SEARCH_Y_MIN >= SEARCH_Y_MAX):
        raise ValueError("invalid grayscale search area")
    if HOUGH_EVERY_N_FRAMES < 1:
        raise ValueError("HOUGH_EVERY_N_FRAMES must be >= 1")
    if DISPLAY_EVERY_N_FRAMES < 1 or PRINT_EVERY_N_FRAMES < 1:
        raise ValueError("display/print interval must be >= 1")
    if UART_SEND_EVERY_N_FRAMES < 1:
        raise ValueError("UART_SEND_EVERY_N_FRAMES must be >= 1")
    if STREAM_WIDTH <= 0 or STREAM_HEIGHT <= 0 or STREAM_FPS < 1:
        raise ValueError("invalid stream dimensions or FPS")
    if STREAM_JPEG_QUALITY < 1 or STREAM_JPEG_QUALITY > 100:
        raise ValueError("STREAM_JPEG_QUALITY must be in 1..100")


def pack_uart_frame(output):
    if output is None:
        return "N\r\n"
    return "%d\r\n" % output[7]


def patch_mean(gray, x0, y0, x1, y1):
    patch = gray[y0:y1, x0:x1]
    value = cv2.mean(patch)
    return float(value[0])


def local_contrast(gray, cx, cy, radius):
    inner_r = max(3, radius)
    outer_r = radius + max(4, radius // 2)
    ix0 = max(0, cx - inner_r)
    iy0 = max(0, cy - inner_r)
    ix1 = min(ROI_W, cx + inner_r + 1)
    iy1 = min(ROI_H, cy + inner_r + 1)
    ox0 = max(0, cx - outer_r)
    oy0 = max(0, cy - outer_r)
    ox1 = min(ROI_W, cx + outer_r + 1)
    oy1 = min(ROI_H, cy + outer_r + 1)
    inner_mean = patch_mean(gray, ix0, iy0, ix1, iy1)
    outer_mean = patch_mean(gray, ox0, oy0, ox1, oy1)
    inner_area = (ix1 - ix0) * (iy1 - iy0)
    outer_area = (ox1 - ox0) * (oy1 - oy0)
    ring_area = outer_area - inner_area
    if ring_area <= 0:
        return 0.0
    ring_mean = (outer_mean * outer_area - inner_mean * inner_area) / ring_area
    return ring_mean - inner_mean


def blackhat_peak(gray, blackhat_kernel, x0, y0, x1, y1, minimum):
    if x1 <= x0 or y1 <= y0:
        return None
    patch = gray[y0:y1, x0:x1]
    blackhat = cv2.morphologyEx(patch, cv2.MORPH_BLACKHAT, blackhat_kernel)
    response = cv2.GaussianBlur(
        blackhat,
        (BLACKHAT_BLUR_KERNEL, BLACKHAT_BLUR_KERNEL),
        0,
    )
    guard = BLACKHAT_KERNEL_SIZE // 2
    patch_w = x1 - x0
    patch_h = y1 - y0
    if patch_w <= guard * 2 or patch_h <= guard * 2:
        return None
    valid_response = response[
        guard:patch_h - guard,
        guard:patch_w - guard,
    ]
    _, max_value, _, max_location = cv2.minMaxLoc(valid_response)
    if max_value < minimum:
        return None
    return (
        x0 + guard + int(max_location[0]),
        y0 + guard + int(max_location[1]),
        float(max_value),
    )


def detect_dark_spot(gray, blackhat_kernel, track):
    """Locate the moving dark feature, searching near prediction first."""
    peak = None
    if track["valid"]:
        predicted_x = int(track["cx"] + track["vx"] + 0.5)
        predicted_y = int(track["cy"] + track["vy"] + 0.5)
        x0 = max(SEARCH_X_MARGIN, predicted_x - TRACK_SEARCH_X)
        x1 = min(ROI_W - SEARCH_X_MARGIN, predicted_x + TRACK_SEARCH_X + 1)
        y0 = max(SEARCH_Y_MIN, predicted_y - TRACK_SEARCH_Y)
        y1 = min(SEARCH_Y_MAX, predicted_y + TRACK_SEARCH_Y + 1)
        peak = blackhat_peak(
            gray, blackhat_kernel, x0, y0, x1, y1, MIN_TRACKED_RESPONSE
        )
    if peak is None:
        peak = blackhat_peak(
            gray,
            blackhat_kernel,
            SEARCH_X_MARGIN,
            SEARCH_Y_MIN,
            ROI_W - SEARCH_X_MARGIN,
            SEARCH_Y_MAX,
            MIN_BLACKHAT_RESPONSE,
        )
    if peak is None:
        return None
    cx = peak[0]
    cy = peak[1] + BLACKHAT_CENTER_Y_OFFSET
    cy = max(0, min(ROI_H - 1, cy))
    return (cx, cy, FALLBACK_RADIUS, peak[2])


def select_circle(gray, circles, anchor, last_track):
    if anchor is None:
        return None
    if circles is None:
        return anchor
    best = None
    best_score = -1000000.0
    count = min(circles.shape[0], MAX_CIRCLES)
    for index in range(count):
        cx = int(circles[index, 0])
        cy = int(circles[index, 1])
        radius = int(circles[index, 2])
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue
        if (cx - radius <= EDGE_MARGIN or cy - radius <= EDGE_MARGIN or
                cx + radius >= ROI_W - EDGE_MARGIN or
                cy + radius >= ROI_H - EDGE_MARGIN):
            continue
        anchor_distance = abs(cx - anchor[0]) + abs(cy - anchor[1])
        if anchor_distance > MAX_ANCHOR_DISTANCE:
            continue
        contrast = local_contrast(gray, cx, cy, radius)
        if contrast < MIN_LOCAL_CONTRAST:
            continue
        score = contrast * 5.0 - anchor_distance
        if last_track["valid"]:
            distance = abs(cx - last_track["cx"]) + abs(cy - last_track["cy"])
            if distance <= MAX_CENTER_JUMP:
                score += MAX_CENTER_JUMP - distance
            else:
                score -= distance
        if score > best_score:
            best_score = score
            best = (cx, cy, radius, anchor[3])
    return best if best is not None else anchor


def update_track(track, candidate):
    if candidate is None:
        if track["valid"]:
            track["lost"] += 1
            if track["lost"] <= LOST_HOLD_FRAMES:
                track["cx"] += track["vx"]
                track["cy"] += track["vy"]
                track["vx"] *= VELOCITY_DECAY
                track["vy"] *= VELOCITY_DECAY
            else:
                track["valid"] = False
                track["hits"] = 0
        return
    cx, cy, radius, strength = candidate
    if track["valid"]:
        predicted_x = track["cx"] + track["vx"]
        predicted_y = track["cy"] + track["vy"]
        distance = abs(cx - predicted_x) + abs(cy - predicted_y)
        if distance <= MAX_CENTER_JUMP:
            measured_vx = cx - track["last_measure_x"]
            measured_vy = cy - track["last_measure_y"]
            velocity_alpha = VELOCITY_ALPHA
            track["vx"] = (
                track["vx"] * (1.0 - velocity_alpha) +
                measured_vx * velocity_alpha
            )
            track["vy"] = (
                track["vy"] * (1.0 - velocity_alpha) +
                measured_vy * velocity_alpha
            )
            speed = abs(measured_vx) + abs(measured_vy)
            position_alpha = FAST_POSITION_ALPHA if speed >= FAST_MOTION_THRESHOLD else POSITION_ALPHA
            track["cx"] += (cx - track["cx"]) * position_alpha
            track["cy"] += (cy - track["cy"]) * position_alpha
            track["radius"] += (radius - track["radius"]) * RADIUS_ALPHA
            track["last_measure_x"] = float(cx)
            track["last_measure_y"] = float(cy)
            track["strength"] = strength
            track["hits"] += 1
            track["lost"] = 0
            return
    track["valid"] = True
    track["cx"] = float(cx)
    track["cy"] = float(cy)
    track["radius"] = float(radius)
    track["strength"] = strength
    track["vx"] = 0.0
    track["vy"] = 0.0
    track["last_measure_x"] = float(cx)
    track["last_measure_y"] = float(cy)
    track["hits"] = 1
    track["lost"] = 0


def current_output(track):
    """Return ROI position, frame position and error from camera center."""
    if not track["valid"] or track["hits"] < CONFIRM_FRAMES:
        return None
    cx = int(track["cx"] + track["vx"] * DISPLAY_PREDICTION + 0.5)
    cy = int(track["cy"] + track["vy"] * DISPLAY_PREDICTION + 0.5)
    cx = max(0, min(ROI_W - 1, cx))
    cy = max(0, min(ROI_H - 1, cy))
    radius = max(1, int(track["radius"] + 0.5))
    frame_x = CAPTURE_ORIGIN_X + ROI_X + cx
    frame_y = CAPTURE_ORIGIN_Y + ROI_Y + cy
    dx = frame_x - CAMERA_CENTER_X
    dy = frame_y - CAMERA_CENTER_Y
    dx_mm_float = dx * BALL_DIAMETER_MM / BALL_DIAMETER_PIXELS
    if dx_mm_float >= 0:
        dx_mm = int(dx_mm_float + 0.5)
    else:
        dx_mm = int(dx_mm_float - 0.5)
    return (cx, cy, radius, frame_x, frame_y, dx, dy, dx_mm)


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
    if len(password) < 8:
        raise ValueError("AP password must contain at least 8 characters")
    ap = network.WLAN(network.AP_IF)
    try:
        ap.stop()
        time.sleep_ms(500)
    except Exception as error:
        print("[AP] previous hotspot stop skipped:", error)
    # Yahboom CanMV v1.8 accepts exactly these two keyword arguments.
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


class APGrayStream:
    """Compress and serve copied detector frames without camera concurrency."""

    def __init__(self):
        self.stream_buffer = image.Image(
            STREAM_WIDTH,
            STREAM_HEIGHT,
            image.GRAYSCALE,
        )
        self.pending_output = None
        self.frame_ready = False
        self.buffer_busy = False
        self.viewer_connected = False
        self.copy_error_reported = False
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
        try:
            _thread.start_new_thread(self._worker, ())
        except Exception:
            self.running = False
            self.thread_done = True
            raise
        ip_address = self.ap.ifconfig()[0]
        print("[READY] Connect Wi-Fi:", AP_SSID)
        print("[READY] Open browser: http://%s:%d/" % (ip_address, HTTP_PORT))
        print("[READY] stream uses decimated copies of CAM_CHN_ID_0")

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
        # RT-Smart networking is always active; do not call active(False).
        self.ap = None

    def offer_frame(self, frame, output, frame_count):
        """Copy at most one low-rate frame; never wait for the HTTP worker."""
        if (not self.running or not self.viewer_connected or
                frame_count % STREAM_FRAME_DIV != 0 or self.buffer_busy):
            return
        self.buffer_busy = True
        try:
            self.stream_buffer.copy_from(frame)
            self.pending_output = output
            self.frame_ready = True
        except Exception as error:
            self.buffer_busy = False
            self.frame_ready = False
            if not self.copy_error_reported:
                print("[HTTP] frame copy error:", error)
                self.copy_error_reported = True

    def _draw_result(self, frame, output):
        if not STREAM_DRAW_RESULT:
            return
        scale_x = STREAM_WIDTH / float(FRAME_WIDTH)
        scale_y = STREAM_HEIGHT / float(FRAME_HEIGHT)
        axis_roi_x = CAMERA_CENTER_X - CAPTURE_ORIGIN_X - ROI_X
        if 0 <= axis_roi_x < ROI_W:
            axis_x = int((ROI_X + axis_roi_x) * scale_x)
            frame.draw_line(
                axis_x, 0, axis_x, STREAM_HEIGHT - 1,
                color=(255, 255, 255), thickness=1,
            )
        if output is None:
            return
        cx, cy, radius = output[0], output[1], output[2]
        x0 = max(0, cx - radius - BOX_MARGIN)
        y0 = max(0, cy - radius - BOX_MARGIN)
        x1 = min(ROI_W, cx + radius + BOX_MARGIN)
        y1 = min(ROI_H, cy + radius + BOX_MARGIN)
        draw_x = int((ROI_X + x0) * scale_x)
        draw_y = int((ROI_Y + y0) * scale_y)
        draw_w = max(1, int((x1 - x0) * scale_x))
        draw_h = max(1, int((y1 - y0) * scale_y))
        frame.draw_rectangle(
            draw_x, draw_y, draw_w, draw_h,
            color=(255, 255, 255), thickness=2,
        )
        center_x = int((ROI_X + cx) * scale_x)
        center_y = int((ROI_Y + cy) * scale_y)
        frame.draw_line(
            center_x - 4, center_y, center_x + 4, center_y,
            color=(255, 255, 255), thickness=1,
        )
        frame.draw_line(
            center_x, center_y - 4, center_x, center_y + 4,
            color=(255, 255, 255), thickness=1,
        )

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
        self.viewer_connected = True
        try:
            while self.running:
                if not self.frame_ready:
                    time.sleep_ms(2)
                    continue
                started = time.ticks_ms()
                frame = self.stream_buffer
                output = self.pending_output
                self._draw_result(frame, output)
                jpeg = frame.compressed(quality=STREAM_JPEG_QUALITY)
                jpeg_size = jpeg.size()
                jpeg_data = uctypes.bytearray_at(jpeg.virtaddr(), jpeg_size)
                part_header = (
                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % jpeg_size
                ).encode("ascii")
                _send_all(client, part_header)
                _send_all(client, jpeg_data)
                _send_all(client, b"\r\n")
                del jpeg_data
                del jpeg
                self.frame_ready = False
                self.buffer_busy = False
                remaining = interval_ms - time.ticks_diff(time.ticks_ms(), started)
                if remaining > 0:
                    time.sleep_ms(remaining)
        finally:
            self.viewer_connected = False
            self.frame_ready = False
            self.buffer_busy = False

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


def main():
    validate_config()

    sensor = None
    uart = None
    fpioa = None
    stream_server = None
    display_initialized = False
    media_initialized = False

    preview_scale = min(
        DISPLAY_WIDTH / float(ROI_W),
        DISPLAY_HEIGHT / float(ROI_H),
    )
    preview_w = int(ROI_W * preview_scale)
    preview_h = int(ROI_H * preview_scale)
    preview_x = (DISPLAY_WIDTH - preview_w) // 2
    preview_y = (DISPLAY_HEIGHT - preview_h) // 2

    canvas_np = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=np.uint8)
    display_image = image.Image(
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        image.GRAYSCALE,
        alloc=image.ALLOC_REF,
        data=canvas_np,
    )
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE),
    )

    track = {
        "valid": False,
        "cx": 0.0,
        "cy": 0.0,
        "radius": 0.0,
        "strength": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "last_measure_x": 0.0,
        "last_measure_y": 0.0,
        "hits": 0,
        "lost": 0,
    }

    try:
        sensor = Sensor(
            id=CAMERA_ID,
            width=SENSOR_WIDTH,
            height=SENSOR_HEIGHT,
            fps=CAMERA_FPS,
        )
        sensor.reset()
        sensor.set_framesize(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            chn=CAM_CHN_ID_0,
            crop=(SENSOR_CROP_X, SENSOR_CROP_Y, SENSOR_CROP_W, SENSOR_CROP_H),
        )
        sensor.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_0)

        Display.init(
            Display.ST7701,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            to_ide=DISPLAY_TO_IDE,
            quality=30,
        )
        display_initialized = True
        MediaManager.init()
        media_initialized = True

        if UART_ENABLED:
            fpioa = FPIOA()
            fpioa.set_function(UART_TX_PIN, FPIOA.UART1_TXD)
            fpioa.set_function(UART_RX_PIN, FPIOA.UART1_RXD)
            uart = UART(UART.UART1, UART_BAUD)

        sensor.run()
        time.sleep_ms(500)

        # AP failure disables only web streaming; tracking and UART continue.
        stream_server = APGrayStream()
        try:
            stream_server.start()
        except Exception as ap_error:
            print("[AP] web stream disabled:", ap_error)
            try:
                stream_server.stop()
            except Exception:
                pass
            stream_server = None

        clock = time.clock()
        frame_count = 0
        print("grayscale ball tracker with AP stream started")
        print("detection: %dx%d ch0, stream: %dx%d ch1 @ %dfps" % (
            FRAME_WIDTH, FRAME_HEIGHT, STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS
        ))
        print("capture: %dx%d sensor_crop=(%d,%d,%d,%d)" % (
            FRAME_WIDTH,
            FRAME_HEIGHT,
            SENSOR_CROP_X,
            SENSOR_CROP_Y,
            SENSOR_CROP_W,
            SENSOR_CROP_H,
        ))
        print("ROI: x=%d y=%d w=%d h=%d reference_origin=(%d,%d)" % (
            ROI_X,
            ROI_Y,
            ROI_W,
            ROI_H,
            CAPTURE_ORIGIN_X,
            CAPTURE_ORIGIN_Y,
        ))
        if uart is not None:
            print("UART1: TX=%d RX=%d baud=%d center_x=%d" %
                  (UART_TX_PIN, UART_RX_PIN, UART_BAUD, CAMERA_CENTER_X))

        while True:
            launcher_control.checkpoint()
            clock.tick()

            frame = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_np = frame.to_numpy_ref()
            roi_gray = frame_np[
                ROI_Y:ROI_Y + ROI_H,
                ROI_X:ROI_X + ROI_W,
            ]
            anchor = detect_dark_spot(roi_gray, blackhat_kernel, track)
            candidate = anchor

            if (ENABLE_HOUGH_REFINEMENT and anchor is not None and
                    frame_count % HOUGH_EVERY_N_FRAMES == 0):
                blurred = cv2.GaussianBlur(
                    roi_gray,
                    (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL),
                    GAUSSIAN_SIGMA,
                )
                circles = cv2.HoughCircles(
                    blurred,
                    3,
                    dp=HOUGH_DP,
                    minDist=HOUGH_MIN_DISTANCE,
                    param1=HOUGH_CANNY_THRESHOLD,
                    param2=HOUGH_SCORE_THRESHOLD,
                    minRadius=MIN_RADIUS,
                    maxRadius=MAX_RADIUS,
                )
                candidate = select_circle(blurred, circles, anchor, track)
            update_track(track, candidate)

            output = current_output(track)
            # A tiny 640x67 copy is attempted only at the stream rate. If the
            # network worker is busy, this frame is dropped instead of waiting.
            if stream_server is not None:
                stream_server.offer_frame(frame, output, frame_count)

            if output is not None:
                cx, cy, radius, center_x, center_y, dx, dy, dx_mm = output
                status = "BALL x:%dmm" % dx_mm
            else:
                cx = -1
                cy = -1
                radius = 0
                center_x = -1
                center_y = -1
                dx = 0
                dy = 0
                dx_mm = 0
                status = "SEARCHING" if track["valid"] else "NO BALL"

            if (uart is not None and
                    frame_count % UART_SEND_EVERY_N_FRAMES == 0):
                try:
                    uart.write(pack_uart_frame(output))
                except Exception as uart_error:
                    print("UART write error:", uart_error)

            if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                preview = cv2.resize(roi_gray, (preview_w, preview_h))
                canvas_np[:, :] = 0
                canvas_np[
                    preview_y:preview_y + preview_h,
                    preview_x:preview_x + preview_w,
                ] = preview

                axis_roi_x = CAMERA_CENTER_X - ROI_X
                if 0 <= axis_roi_x < ROI_W:
                    axis_x = preview_x + int(axis_roi_x * preview_scale)
                    display_image.draw_line(
                        axis_x,
                        preview_y,
                        axis_x,
                        preview_y + preview_h - 1,
                        color=AXIS_COLOR,
                        thickness=1,
                    )

                if output is not None:
                    x0 = max(0, cx - radius - BOX_MARGIN)
                    y0 = max(0, cy - radius - BOX_MARGIN)
                    x1 = min(ROI_W, cx + radius + BOX_MARGIN)
                    y1 = min(ROI_H, cy + radius + BOX_MARGIN)
                    draw_x = preview_x + int(x0 * preview_scale)
                    draw_y = preview_y + int(y0 * preview_scale)
                    draw_w = max(1, int((x1 - x0) * preview_scale))
                    draw_h = max(1, int((y1 - y0) * preview_scale))
                    display_image.draw_rectangle(
                        draw_x,
                        draw_y,
                        draw_w,
                        draw_h,
                        color=(0, 0, 0),
                        thickness=2,
                    )

                display_image.draw_string_advanced(
                    8, 8, 24, status, color=(255, 255, 255)
                )
                Display.show_image(display_image)

            frame_count += 1
            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                print(
                    "ball=(%d,%d) x=%dmm err_px=(%d,%d) v=(%.1f,%.1f) resp=%.1f fps=%.1f" %
                    (
                        center_x,
                        center_y,
                        dx_mm,
                        dx,
                        dy,
                        track["vx"],
                        track["vy"],
                        track["strength"],
                        clock.fps(),
                    )
                )
                gc.collect()

    except KeyboardInterrupt:
        print("user stopped")
    except BaseException as error:
        print("error:", error)
        raise
    finally:
        if stream_server is not None:
            try:
                stream_server.stop()
            except Exception as stream_error:
                print("stream stop error:", stream_error)
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        if uart is not None:
            try:
                uart.deinit()
            except Exception:
                pass
        if display_initialized:
            Display.deinit()
        launcher_control.checkpoint(0)
        time.sleep_ms(100)
        if media_initialized:
            MediaManager.deinit()


if __name__ == "__main__":
    main()
