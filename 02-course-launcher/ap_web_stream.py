"""Yahboom K230 AP-mode browser video streaming with local recording.

The K230 creates its own Wi-Fi hotspot and serves an MJPEG stream over HTTP.
After a computer connects to the hotspot, open the printed URL in Chrome or
Edge. Recording is performed in the browser and downloaded as a real MP4 file, so
recording does not consume the K230 SD card.
"""

import gc
import socket
import time
import network
import uctypes

import launcher_control

from media.media import MediaManager
from media.sensor import Sensor, CAM_CHN_ID_1


# Hotspot settings. The password must contain at least 8 characters.
AP_SSID = "K230-CAM"
AP_PASSWORD = "YOUR_AP_PASSWORD"

# HTTP and camera settings. Lower JPEG_QUALITY or TARGET_FPS if transmission
# is unstable. 640x480 @ 10 fps is a conservative MJPEG starting point.
HTTP_PORT = 8081
SENSOR_ID = 2
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
TARGET_FPS = 10
JPEG_QUALITY = 55
CLIENT_TIMEOUT_SECONDS = 8
SOCKET_RETRY_DELAY_MS = 10
REQUEST_TIMEOUT_MS = 5000

_BOUNDARY = b"frame"
_CRLF = b"\r\n"


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <title>K230 AP 实时图传</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070b14;
      --panel: rgba(18, 27, 46, .88);
      --line: rgba(148, 163, 184, .22);
      --text: #eef4ff;
      --muted: #94a3b8;
      --blue: #38bdf8;
      --green: #34d399;
      --red: #fb7185;
      --yellow: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 0%, rgba(14, 165, 233, .18), transparent 32rem),
        radial-gradient(circle at 100% 100%, rgba(52, 211, 153, .12), transparent 28rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, "Microsoft YaHei", system-ui, sans-serif;
    }
    .page { width: min(1100px, 100%); margin: 0 auto; padding: 22px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    h1 { margin: 0; font-size: clamp(22px, 4vw, 34px); letter-spacing: -.03em; }
    .subtitle { margin-top: 7px; color: var(--muted); font-size: 14px; }
    .live {
      display: inline-flex; align-items: center; gap: 8px;
      border: 1px solid var(--line); border-radius: 999px;
      background: rgba(15, 23, 42, .75); padding: 8px 12px;
      color: var(--muted); font-size: 13px; white-space: nowrap;
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--yellow); box-shadow: 0 0 14px currentColor; }
    .live.ready .dot { background: var(--green); }
    .live.recording .dot { background: var(--red); animation: pulse 1s infinite; }
    @keyframes pulse { 50% { opacity: .28; } }
    .viewer {
      overflow: hidden; position: relative; min-height: 260px;
      background: #000; border: 1px solid var(--line); border-radius: 18px;
      box-shadow: 0 24px 70px rgba(0, 0, 0, .36);
    }
    #stream { display: block; width: 100%; height: auto; min-height: 260px; object-fit: contain; background: #000; }
    .overlay {
      position: absolute; left: 14px; bottom: 14px;
      display: flex; align-items: center; gap: 8px;
      padding: 7px 10px; border-radius: 10px;
      background: rgba(2, 6, 23, .7); backdrop-filter: blur(8px);
      font: 12px ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    #recordCanvas { position: fixed; width: 2px; height: 2px; left: -20px; top: -20px; opacity: .01; pointer-events: none; }
    .panel {
      margin-top: 16px; padding: 16px; border-radius: 16px;
      border: 1px solid var(--line); background: var(--panel);
    }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, .download {
      appearance: none; border: 1px solid var(--line); border-radius: 11px;
      padding: 11px 16px; font: inherit; font-weight: 700; cursor: pointer;
      color: var(--text); background: #1e293b; text-decoration: none;
      transition: transform .12s ease, border-color .12s ease, opacity .12s ease;
    }
    button:hover, .download:hover { transform: translateY(-1px); border-color: rgba(56, 189, 248, .65); }
    button:disabled, .download.disabled { opacity: .38; cursor: not-allowed; transform: none; pointer-events: none; }
    #startRecord { color: #022c22; background: var(--green); }
    #stopRecord { color: white; background: #be123c; }
    #snapshot { color: #082f49; background: var(--blue); }
    .meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
    .item { padding: 11px 12px; border: 1px solid var(--line); border-radius: 11px; background: rgba(2, 6, 23, .38); }
    .label { color: var(--muted); font-size: 12px; }
    .value { margin-top: 4px; font-weight: 750; }
    .note { color: var(--muted); font-size: 13px; line-height: 1.65; margin: 14px 0 0; }
    @media (max-width: 650px) {
      .page { padding: 14px; }
      header { align-items: flex-start; flex-direction: column; }
      .meta { grid-template-columns: 1fr; }
      button, .download { flex: 1 1 calc(50% - 10px); text-align: center; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>K230 AP 实时图传</h1>
        <div class="subtitle">浏览器直连 · 录像保存在电脑本地</div>
      </div>
      <div id="state" class="live"><span class="dot"></span><span id="stateText">正在连接视频…</span></div>
    </header>

    <section class="viewer">
      <img id="stream" src="/stream.mjpg" alt="K230 实时视频">
      <div class="overlay"><span id="resolution">等待画面</span><span>·</span><span id="timer">00:00</span></div>
    </section>

    <section class="panel">
      <div class="controls">
        <button id="startRecord" disabled>开始录像</button>
        <button id="stopRecord" disabled>停止录像</button>
        <a id="download" class="download disabled">下载录像</a>
        <button id="snapshot" disabled>保存截图</button>
      </div>
      <div class="meta">
        <div class="item"><div class="label">传输方式</div><div class="value">HTTP MJPEG</div></div>
        <div class="item"><div class="label">录像格式</div><div id="recordFormat" class="value">检测中…</div></div>
        <div class="item"><div class="label">录像位置</div><div class="value">浏览器下载目录</div></div>
      </div>
      <p id="message" class="note">建议使用最新版 Chrome 或 Edge。录像由电脑浏览器编码为 MP4，不写入 K230 存储卡。</p>
    </section>

    <canvas id="recordCanvas"></canvas>
  </main>

  <script>
    (() => {
      const stream = document.getElementById('stream');
      const canvas = document.getElementById('recordCanvas');
      const ctx = canvas.getContext('2d', { alpha: false });
      const state = document.getElementById('state');
      const stateText = document.getElementById('stateText');
      const resolution = document.getElementById('resolution');
      const timer = document.getElementById('timer');
      const message = document.getElementById('message');
      const startButton = document.getElementById('startRecord');
      const stopButton = document.getElementById('stopRecord');
      const snapshotButton = document.getElementById('snapshot');
      const download = document.getElementById('download');
      const recordFormat = document.getElementById('recordFormat');

      let recorder = null;
      let chunks = [];
      let recordUrl = null;
      let recordStartedAt = 0;
      let timerId = null;
      let drawing = false;

      const mimeCandidates = [
        'video/mp4;codecs=avc1.42E01E',
        'video/mp4;codecs=avc1',
        'video/mp4;codecs=h264',
        'video/mp4'
      ];
      const hasMediaRecorder = typeof window.MediaRecorder !== 'undefined';
      const hasMimeProbe = hasMediaRecorder && typeof MediaRecorder.isTypeSupported === 'function';
      const mimeType = hasMimeProbe
        ? (mimeCandidates.find(type => MediaRecorder.isTypeSupported(type)) || '')
        : '';
      recordFormat.textContent = mimeType || '不支持 MP4/H.264';

      function filename(prefix, extension) {
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        return `${prefix}-${stamp}.${extension}`;
      }

      function drawFrame() {
        if (stream.naturalWidth && stream.naturalHeight) {
          if (canvas.width !== stream.naturalWidth || canvas.height !== stream.naturalHeight) {
            canvas.width = stream.naturalWidth;
            canvas.height = stream.naturalHeight;
          }
          ctx.drawImage(stream, 0, 0, canvas.width, canvas.height);
        }
        requestAnimationFrame(drawFrame);
      }

      function setReady() {
        if (!drawing) {
          drawing = true;
          requestAnimationFrame(drawFrame);
        }
        resolution.textContent = `${stream.naturalWidth} × ${stream.naturalHeight}`;
        state.className = 'live ready';
        stateText.textContent = '实时画面已连接';
        const canRecordMp4 = hasMediaRecorder && typeof canvas.captureStream === 'function' && Boolean(mimeType);
        startButton.disabled = !canRecordMp4;
        snapshotButton.disabled = false;
        if (!hasMediaRecorder || typeof canvas.captureStream !== 'function') {
          message.textContent = '当前浏览器不支持网页录像，请改用最新版 Chrome 或 Edge。';
        } else if (!mimeType) {
          message.textContent = '当前浏览器不能直接编码 MP4/H.264，录像按钮已禁用。请升级到最新版 Chrome 或 Edge。';
        } else {
          message.textContent = `已启用 MP4 录像：${mimeType}。录像由电脑浏览器编码，不写入 K230 存储卡。`;
        }
      }

      stream.addEventListener('load', setReady);
      stream.addEventListener('error', () => {
        if (recorder && recorder.state === 'recording') recorder.stop();
        state.className = 'live';
        stateText.textContent = '视频断开，正在重连…';
        startButton.disabled = true;
        snapshotButton.disabled = true;
        window.setTimeout(() => {
          stream.src = `/stream.mjpg?t=${Date.now()}`;
        }, 1500);
      });

      function updateTimer() {
        const seconds = Math.floor((Date.now() - recordStartedAt) / 1000);
        const minutes = String(Math.floor(seconds / 60)).padStart(2, '0');
        const rest = String(seconds % 60).padStart(2, '0');
        timer.textContent = `${minutes}:${rest}`;
      }

      startButton.addEventListener('click', () => {
        if (!hasMediaRecorder || typeof canvas.captureStream !== 'function' || !mimeType) {
          message.textContent = '无法开始录像：浏览器必须支持 MP4/H.264 MediaRecorder。请使用最新版 Chrome 或 Edge。';
          return;
        }
        if (recordUrl) {
          URL.revokeObjectURL(recordUrl);
          recordUrl = null;
        }
        download.removeAttribute('href');
        download.classList.add('disabled');
        chunks = [];
        const mediaStream = canvas.captureStream(15);
        const options = { mimeType, videoBitsPerSecond: 4000000 };
        try {
          recorder = new MediaRecorder(mediaStream, options);
        } catch (error) {
          mediaStream.getTracks().forEach(track => track.stop());
          startButton.disabled = false;
          stopButton.disabled = true;
          message.textContent = `MP4 录像器创建失败：${error.message || error}`;
          return;
        }
        recorder.addEventListener('dataavailable', event => {
          if (event.data && event.data.size) chunks.push(event.data);
        });
        recorder.addEventListener('stop', () => {
          clearInterval(timerId);
          timerId = null;
          const finalType = recorder.mimeType || mimeType;
          const blob = new Blob(chunks, { type: finalType });
          recordUrl = URL.createObjectURL(blob);
          download.href = recordUrl;
          download.download = filename('k230-recording', 'mp4');
          download.classList.remove('disabled');
          state.className = 'live ready';
          stateText.textContent = '录像完成，可下载';
          startButton.disabled = false;
          stopButton.disabled = true;
          message.textContent = `录像已生成：${(blob.size / 1024 / 1024).toFixed(1)} MB，格式为 MP4。点击“下载录像”保存。`;
          mediaStream.getTracks().forEach(track => track.stop());
        });
        recorder.start(1000);
        recordStartedAt = Date.now();
        updateTimer();
        timerId = setInterval(updateTimer, 250);
        state.className = 'live recording';
        stateText.textContent = '正在录像';
        startButton.disabled = true;
        stopButton.disabled = false;
        message.textContent = '正在电脑内存中录像。请先点击“停止录像”，再断开热点或关闭页面。';
      });

      stopButton.addEventListener('click', () => {
        if (recorder && recorder.state !== 'inactive') recorder.stop();
        stopButton.disabled = true;
      });

      snapshotButton.addEventListener('click', () => {
        if (!stream.naturalWidth) return;
        ctx.drawImage(stream, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename('k230-snapshot', 'jpg');
          link.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }, 'image/jpeg', .92);
      });

      window.addEventListener('beforeunload', () => {
        if (recordUrl) URL.revokeObjectURL(recordUrl);
      });
    })();
  </script>
</body>
</html>
"""


def _exitpoint():
    """Check the launcher return key and the firmware cooperative hook."""
    launcher_control.checkpoint()


def _get_ap_info(ap):
    """Read AP details across old and new CanMV network APIs."""
    try:
        return ap.info()
    except Exception:
        try:
            return ap.config("info")
        except Exception as error:
            return "unavailable: %s" % error


def create_ap(ssid, password):
    """Create a WPA hotspot on the always-active RT-Smart interface."""
    if len(password) < 8:
        raise ValueError("AP password must contain at least 8 characters")

    ap = network.WLAN(network.AP_IF)

    # Stop a previously running AP first. This clears stale SSID/security
    # settings after rerunning a script in CanMV IDE, but does not disable the
    # always-active RT-Smart network subsystem.
    try:
        ap.stop()
        time.sleep_ms(500)
    except Exception as error:
        print("[AP] previous hotspot stop skipped:", error)

    # This Yahboom CanMV v1.8 build accepts exactly ssid= and key= for AP
    # creation. Supplying password= or channel= raises "extra keyword
    # arguments given", so keep the call identical to Yahboom's ap.py.
    ap.config(ssid=ssid, key=password)

    # Give hostapd/DHCP enough time to publish the new security settings.
    time.sleep(3)
    print("[AP] SSID:", ssid)
    print("[AP] password:", password)
    print("[AP] configured with field: key")
    print("[AP] info:", _get_ap_info(ap))
    print("[AP] ifconfig:", ap.ifconfig())
    return ap


def _error_number(error):
    number = getattr(error, "errno", None)
    if number is None:
        try:
            number = error.args[0]
        except Exception:
            number = None
    return number


def _would_block(error):
    # 11 is EAGAIN on RT-Smart. 35 and 10035 make local tests portable.
    return _error_number(error) in (11, 35, 10035)


def _send_all(sock, data):
    """Write all bytes, retrying RT-Smart's transient EAGAIN result."""
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
            if time.ticks_diff(time.ticks_ms(), last_progress) >= CLIENT_TIMEOUT_SECONDS * 1000:
                raise OSError("socket send timed out")
            _exitpoint()
            time.sleep_ms(SOCKET_RETRY_DELAY_MS)
            continue

        if sent <= 0:
            raise OSError("socket closed during send")

        sent_total += sent
        last_progress = time.ticks_ms()


def _send_response(client, status, content_type, body=b"", extra_headers=b""):
    if isinstance(body, str):
        body = body.encode("utf-8")
    header = (
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Connection: close\r\n" % (status, content_type, len(body))
    ).encode("utf-8")
    _send_all(client, header)
    if extra_headers:
        _send_all(client, extra_headers)
    _send_all(client, _CRLF)
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
            if time.ticks_diff(time.ticks_ms(), last_progress) >= REQUEST_TIMEOUT_MS:
                raise OSError("HTTP request timed out")
            _exitpoint()
            time.sleep_ms(SOCKET_RETRY_DELAY_MS)
            continue

        if not chunk:
            break

        request.extend(chunk)
        last_progress = time.ticks_ms()
        if b"\r\n\r\n" in request:
            break
        _exitpoint()

    if not request:
        return "/"

    first_line = bytes(request).split(b"\r\n", 1)[0]
    parts = first_line.split()
    if len(parts) < 2:
        return "/"
    try:
        return parts[1].decode("utf-8")
    except Exception:
        return "/"


def _stream_mjpeg(client, sensor):
    header = (
        b"HTTP/1.1 200 OK\r\n"
        b"Age: 0\r\n"
        b"Cache-Control: no-cache, private\r\n"
        b"Pragma: no-cache\r\n"
        b"Connection: close\r\n"
        b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
    )
    _send_all(client, header)
    print("[HTTP] MJPEG client connected")

    frame_interval_ms = max(1, 1000 // TARGET_FPS)
    frame_count = 0
    report_started = time.ticks_ms()

    while True:
        frame_started = time.ticks_ms()
        _exitpoint()

        frame = sensor.snapshot(chn=CAM_CHN_ID_1)
        jpeg = frame.compressed(quality=JPEG_QUALITY)
        jpeg_size = jpeg.size()
        jpeg_data = uctypes.bytearray_at(jpeg.virtaddr(), jpeg_size)

        part_header = (
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %d\r\n\r\n" % jpeg_size
        ).encode("ascii")
        _send_all(client, part_header)
        _send_all(client, jpeg_data)
        _send_all(client, _CRLF)

        frame_count += 1
        del jpeg_data
        del jpeg
        del frame

        now = time.ticks_ms()
        if time.ticks_diff(now, report_started) >= 5000:
            elapsed = time.ticks_diff(now, report_started)
            fps = frame_count * 1000.0 / elapsed if elapsed > 0 else 0.0
            print("[HTTP] stream fps: %.1f" % fps)
            frame_count = 0
            report_started = now
            gc.collect()

        spent = time.ticks_diff(time.ticks_ms(), frame_started)
        remaining = frame_interval_ms - spent
        if remaining > 0:
            time.sleep_ms(remaining)


def _serve_client(client, address, sensor):
    try:
        client.setblocking(False)
        path = _read_request_path(client)
        path_without_query = path.split("?", 1)[0]
        print("[HTTP]", address, path_without_query)

        if path_without_query == "/" or path_without_query == "/index.html":
            _send_response(client, "200 OK", "text/html; charset=utf-8", INDEX_HTML)
        elif path_without_query == "/stream.mjpg":
            _stream_mjpeg(client, sensor)
        elif path_without_query == "/favicon.ico":
            _send_response(client, "204 No Content", "image/x-icon")
        elif path_without_query == "/health":
            _send_response(client, "200 OK", "application/json", b'{"ok":true}')
        else:
            _send_response(client, "404 Not Found", "text/plain; charset=utf-8", "404 Not Found")
    except (OSError, KeyboardInterrupt) as error:
        print("[HTTP] client disconnected:", error)
        if isinstance(error, KeyboardInterrupt):
            raise
    except Exception as error:
        print("[HTTP] client error:", error)
    finally:
        try:
            client.close()
        except Exception:
            pass
        gc.collect()


def main():
    ap = None
    sensor = None
    server = None
    media_initialized = False

    try:
        ap = create_ap(AP_SSID, AP_PASSWORD)

        sensor = Sensor(
            id=SENSOR_ID,
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            fps=TARGET_FPS,
        )
        sensor.reset()
        sensor.set_framesize(
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            chn=CAM_CHN_ID_1,
        )
        sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)

        MediaManager.init()
        media_initialized = True
        sensor.run()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", HTTP_PORT))
        server.listen(3)

        ip_address = ap.ifconfig()[0]
        print("")
        print("[READY] 1. Connect the computer to Wi-Fi:", AP_SSID)
        print("[READY] 2. Open browser: http://%s:%d/" % (ip_address, HTTP_PORT))
        print("[READY] One live-view browser is supported at a time.")

        # RT-Smart sockets are non-blocking and report EAGAIN while no client
        # is waiting. Treat that condition as idle rather than as a fatal error.
        server.setblocking(False)
        while True:
            _exitpoint()
            try:
                client, address = server.accept()
            except OSError as error:
                if not _would_block(error):
                    raise
                time.sleep_ms(SOCKET_RETRY_DELAY_MS)
                continue
            _serve_client(client, address, sensor)

    except KeyboardInterrupt:
        print("[MAIN] stopped by user")
    except BaseException as error:
        print("[MAIN] error:", error)
    finally:
        if server is not None:
            try:
                server.close()
            except Exception:
                pass

        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass

        if media_initialized:
            try:
                MediaManager.deinit()
            except Exception:
                pass

        if ap is not None:
            try:
                ap.stop()
                print("[AP] hotspot stopped")
            except Exception as error:
                print("[AP] hotspot stop skipped:", error)
            ap = None

        gc.collect()
        print("[MAIN] resources released")


if __name__ == "__main__":
    main()
