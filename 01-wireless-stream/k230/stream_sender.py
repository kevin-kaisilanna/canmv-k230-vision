"""K230 hardware H.265 encoder and framed TCP sender.

The media setup follows the Yahboom/CanMV vencoder example. Network writes
are completed explicitly because socket.send() is allowed to write only part
of a buffer, especially over Wi-Fi.
"""

import socket
import time
import uctypes
import ustruct

from media.media import MediaManager
from media.sensor import Sensor
from media.vencoder import (
    Encoder,
    ChnAttrStr,
    StreamData,
    VIDEO_ENCODE_MOD_ID,
    VENC_DEV_ID,
)


DATA_TYPE_CONFIG = 0xFFFFFFFF
DATA_TYPE_VIDEO_FRAME = 0x00000000


class StreamSender:
    def __init__(self, sensor, cam_chn_id, enc_chn_id, width, height,
                 host_ip, port=6001, connect_timeout=5):
        self.sensor = sensor
        self.cam_chn_id = cam_chn_id
        self.enc_chn_id = enc_chn_id
        self.width = width
        self.height = height
        self.host_ip = host_ip
        self.port = port
        self.connect_timeout = connect_timeout
        self.sock = None
        self.encoder_started = False
        self.stream_data = StreamData()

        sensor.set_framesize(
            chn=cam_chn_id,
            width=width,
            height=height,
        )
        sensor.set_pixformat(
            chn=cam_chn_id,
            pix_format=Sensor.YUV420SP,
        )

        MediaManager.link(
            sensor.bind_info(chn=cam_chn_id)["src"],
            (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, enc_chn_id),
        )

        self.encoder = Encoder()
        self.encoder.SetOutBufs(
            chn=enc_chn_id,
            buf_num=8,
            width=width,
            height=height,
        )
        self.channel_attributes = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H265,
            self.encoder.H265_PROFILE_MAIN,
            width,
            height,
        )

    @property
    def connected(self):
        return self.sock is not None

    @staticmethod
    def _send_all(sock, data):
        view = memoryview(data)
        sent_total = 0
        while sent_total < len(view):
            sent = sock.send(view[sent_total:])
            if sent is None:
                # Some MicroPython ports return None after accepting all data.
                return
            if sent <= 0:
                raise OSError("TCP connection closed during send")
            sent_total += sent

    def start(self):
        self.encoder.Create(self.enc_chn_id, self.channel_attributes)
        self.encoder.Start(self.enc_chn_id)
        self.encoder_started = True
        self.connect()

    def connect(self):
        self._close_socket()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host_ip, self.port))
            self._send_all(
                sock,
                ustruct.pack(
                    "<LLL",
                    DATA_TYPE_CONFIG,
                    self.width,
                    self.height,
                ),
            )
            # Frame transmission may take longer on a congested network.
            sock.settimeout(10)
            self.sock = sock
            print("Connected to PC receiver")
            return True
        except Exception as error:
            sock.close()
            print("PC connection failed:", error)
            return False

    def send_frame(self):
        if self.sock is None:
            return False

        stream_acquired = False
        try:
            self.encoder.GetStream(self.enc_chn_id, self.stream_data)
            stream_acquired = True
            pack_count = self.stream_data.pack_cnt
            if pack_count == 0:
                return False

            total_size = 0
            for index in range(pack_count):
                total_size += 4 + self.stream_data.data_size[index]

            self._send_all(
                self.sock,
                ustruct.pack(
                    "<LLL",
                    DATA_TYPE_VIDEO_FRAME,
                    total_size,
                    pack_count,
                ),
            )

            # Send each encoder pack directly to avoid a second full-frame copy.
            for index in range(pack_count):
                nalu_size = self.stream_data.data_size[index]
                self._send_all(self.sock, ustruct.pack("<L", nalu_size))
                nalu_data = uctypes.bytearray_at(
                    self.stream_data.data[index],
                    nalu_size,
                )
                self._send_all(self.sock, nalu_data)
            return True
        except Exception as error:
            print("Frame send failed:", error)
            self._close_socket()
            return False
        finally:
            if stream_acquired:
                self.encoder.ReleaseStream(
                    self.enc_chn_id,
                    self.stream_data,
                )

    def _close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def close(self):
        self._close_socket()
        if self.encoder_started:
            try:
                self.encoder.Stop(self.enc_chn_id)
                self.encoder.Destroy(self.enc_chn_id)
            finally:
                self.encoder_started = False
