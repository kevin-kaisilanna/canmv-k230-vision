"""Yahboom K230 wireless camera streaming based on 07.rtsp/rtsp_no_ai.py.

Open the URL printed by this program with VLC on a PC or phone connected to
the same Wi-Fi network.
"""

import _thread
import gc
import network
import time
import uctypes

import launcher_control
import multimedia as mm
from media.media import *
from media.sensor import *
from media.vencoder import *


# Network settings. Change these when the hotspot changes.
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"
WIFI_TIMEOUT_MS = 20000

# Stream settings. The width must be aligned to 16 pixels.
RTSP_SESSION = "live"
RTSP_PORT = 8554
STREAM_WIDTH = 640
STREAM_HEIGHT = 360


def connect_wifi(ssid, password, timeout_ms=WIFI_TIMEOUT_MS):
    sta = network.WLAN(0)
    if sta.isconnected():
        print("[Wi-Fi] already connected, IP:", sta.ifconfig()[0])
        return sta

    print("[Wi-Fi] connecting to:", ssid)
    sta.connect(ssid, password)
    started = time.ticks_ms()

    while sta.ifconfig()[0] == "0.0.0.0":
        if time.ticks_diff(time.ticks_ms(), started) >= timeout_ms:
            raise RuntimeError("Wi-Fi connection timed out")
        launcher_control.checkpoint()
        time.sleep_ms(200)

    print("[Wi-Fi] connected, IP:", sta.ifconfig()[0])
    return sta


class RtspCameraServer:
    def __init__(self, session_name=RTSP_SESSION, port=RTSP_PORT):
        self.session_name = session_name
        self.port = port
        self.server = mm.rtsp_server()
        self.sensor = None
        self.encoder = None
        self.link = None
        self.running = False
        self.thread_finished = True
        self.encoder_created = False
        self.encoder_started = False
        self.sensor_started = False
        self.rtsp_initialized = False
        self.rtsp_started = False
        self.last_error = None

    def start(self):
        self._init_stream()

        self.server.rtspserver_init(self.port)
        self.rtsp_initialized = True
        self.server.rtspserver_createsession(
            self.session_name,
            mm.multi_media_type.media_h264,
            False,
        )
        self.server.rtspserver_start()
        self.rtsp_started = True

        self.encoder.Start()
        self.encoder_started = True
        self.sensor.run()
        self.sensor_started = True

        self.running = True
        self.thread_finished = False
        _thread.start_new_thread(self._stream_loop, ())

    def _init_stream(self):
        width = ALIGN_UP(STREAM_WIDTH, 16)

        self.sensor = Sensor()
        self.sensor.reset()
        self.sensor.set_framesize(
            width=width,
            height=STREAM_HEIGHT,
            alignment=12,
        )
        self.sensor.set_pixformat(Sensor.YUV420SP)

        self.encoder = Encoder()
        self.encoder.SetOutBufs(8, width, STREAM_HEIGHT)

        attributes = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H264,
            self.encoder.H264_PROFILE_MAIN,
            width,
            STREAM_HEIGHT,
        )
        self.encoder.Create(attributes)
        self.encoder_created = True

        # Current firmware allocates a VENC channel in Create(); bind the
        # camera to that dynamic channel instead of using VENC_CHN_ID_0.
        self.link = MediaManager.link(
            self.sensor.bind_info()["src"],
            (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, self.encoder.chn),
        )

    def _stream_loop(self):
        stream_data = StreamData()
        try:
            while self.running:
                stream_acquired = False
                try:
                    launcher_control.checkpoint()
                    self.encoder.GetStream(stream_data)
                    stream_acquired = True

                    for index in range(stream_data.pack_cnt):
                        size = stream_data.data_size[index]
                        encoded = bytes(
                            uctypes.bytearray_at(stream_data.data[index], size)
                        )
                        self.server.rtspserver_sendvideodata(
                            self.session_name,
                            encoded,
                            size,
                            1000,
                        )
                finally:
                    if stream_acquired:
                        self.encoder.ReleaseStream(stream_data)

        except BaseException as error:
            self.last_error = error
            print("[RTSP] stream stopped:", error)
        finally:
            self.running = False
            self.thread_finished = True

    def get_url(self):
        return self.server.rtspserver_getrtspurl(self.session_name)

    def stop(self):
        self.running = False
        wait_started = time.ticks_ms()
        while (not self.thread_finished and
               time.ticks_diff(time.ticks_ms(), wait_started) < 3000):
            time.sleep_ms(50)

        if self.sensor_started:
            try:
                self.sensor.stop()
            except Exception:
                pass
            self.sensor_started = False

        if self.link is not None:
            self.link = None
            gc.collect()

        if self.encoder_started:
            try:
                self.encoder.Stop()
            except Exception:
                pass
            self.encoder_started = False

        if self.encoder_created:
            try:
                self.encoder.Destroy()
            except Exception:
                pass
            self.encoder_created = False

        if self.rtsp_started:
            try:
                self.server.rtspserver_stop()
            except Exception:
                pass
            self.rtsp_started = False

        if self.rtsp_initialized:
            try:
                self.server.rtspserver_deinit()
            except Exception:
                pass
            self.rtsp_initialized = False


def main():
    wifi = None
    rtsp = None

    try:
        wifi = connect_wifi(WIFI_SSID, WIFI_PASSWORD)
        rtsp = RtspCameraServer()
        rtsp.start()

        # get_url() is kept from the official example. The explicit URL is
        # also printed because some firmware versions return 0.0.0.0.
        print("[RTSP] firmware URL:", rtsp.get_url())
        print(
            "[RTSP] open in VLC: rtsp://%s:%d/%s" %
            (wifi.ifconfig()[0], RTSP_PORT, RTSP_SESSION)
        )

        while rtsp.running:
            launcher_control.checkpoint()
            time.sleep_ms(100)

        if rtsp.last_error is not None:
            raise rtsp.last_error

    except KeyboardInterrupt:
        print("[RTSP] stopped by user")
    except BaseException as error:
        print("[ERROR]", error)
    finally:
        if rtsp is not None:
            rtsp.stop()
        wifi = None
        gc.collect()


if __name__ == "__main__":
    main()
