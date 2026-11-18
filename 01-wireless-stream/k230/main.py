"""Yahboom K230 wireless H.265 camera streaming entry point.

Copy this file and stream_sender.py to /sdcard on the K230. Edit the
three network settings below, start the PC receiver, and then run this file.
"""

import gc
import time
import network

from media.media import MediaManager
from media.sensor import Sensor, CAM_CHN_ID_1
from media.vencoder import VENC_CHN_ID_0

from stream_sender import StreamSender


# Required network settings.
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"
PC_IP = "192.168.43.210"
PC_PORT = 6001

# These defaults favor low latency and stable Wi-Fi transmission.
SENSOR_ID = 2
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
STREAM_FPS = 30
WIFI_TIMEOUT_MS = 20000


def connect_wifi(ssid, password, timeout_ms=WIFI_TIMEOUT_MS):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if not wlan.isconnected():
        print("Connecting to Wi-Fi:", ssid)
        wlan.connect(ssid, password)
        started = time.ticks_ms()
        while not wlan.isconnected():
            if time.ticks_diff(time.ticks_ms(), started) >= timeout_ms:
                raise RuntimeError("Wi-Fi connection timed out")
            time.sleep_ms(200)

    print("Wi-Fi connected, K230 IP:", wlan.ifconfig()[0])
    return wlan


def main():
    wlan = None
    sensor = None
    sender = None
    media_initialized = False

    try:
        wlan = connect_wifi(WIFI_SSID, WIFI_PASSWORD)

        sensor = Sensor(
            id=SENSOR_ID,
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            fps=STREAM_FPS,
        )
        sensor.reset()

        sender = StreamSender(
            sensor=sensor,
            cam_chn_id=CAM_CHN_ID_1,
            enc_chn_id=VENC_CHN_ID_0,
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            host_ip=PC_IP,
            port=PC_PORT,
        )

        MediaManager.init()
        media_initialized = True
        sender.start()
        sensor.run()

        clock = time.clock()
        report_at = time.ticks_add(time.ticks_ms(), 3000)
        print("Streaming to %s:%d" % (PC_IP, PC_PORT))

        while True:
            clock.tick()

            if not sender.connected:
                print("PC is unavailable; retrying in 2 seconds")
                time.sleep_ms(2000)
                sender.connect()
                continue

            sender.send_frame()

            now = time.ticks_ms()
            if time.ticks_diff(now, report_at) >= 0:
                print("stream fps: %.1f" % clock.fps())
                report_at = time.ticks_add(now, 3000)

    except KeyboardInterrupt:
        print("Streaming stopped by user")
    except Exception as error:
        print("Streaming error:", error)
    finally:
        if sender is not None:
            sender.close()
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        if media_initialized:
            MediaManager.deinit()
        wlan = None
        gc.collect()


if __name__ == "__main__":
    main()
