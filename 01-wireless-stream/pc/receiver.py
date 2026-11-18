"""Receive, decode, and display the Yahboom K230 H.265 TCP stream."""

import argparse
import queue
import socket
import struct
import threading
import time

import av
import cv2


DATA_TYPE_CONFIG = 0xFFFFFFFF
DATA_TYPE_VIDEO_FRAME = 0x00000000
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_NALU_COUNT = 128


class ProtocolError(Exception):
    pass


def recv_exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("K230 closed the connection")
        offset += received
    return data


class H265Decoder:
    def __init__(self):
        self.codec = None
        self.width = 0
        self.height = 0
        self.frames = queue.Queue(maxsize=2)

    def configure(self, width, height):
        if not (0 < width <= 4096 and 0 < height <= 4096):
            raise ProtocolError(f"invalid video size: {width}x{height}")
        self.width = width
        self.height = height
        self.codec = av.CodecContext.create("hevc", "r")
        self._clear_frames()
        print(f"Video configured: {width}x{height}, codec=H.265")

    def decode_frame(self, payload, nalu_count):
        if self.codec is None:
            raise ProtocolError("video frame arrived before configuration")

        offset = 0
        for _ in range(nalu_count):
            if offset + 4 > len(payload):
                raise ProtocolError("truncated NALU length")
            nalu_size = struct.unpack_from("<L", payload, offset)[0]
            offset += 4
            end = offset + nalu_size
            if end > len(payload):
                raise ProtocolError("truncated NALU data")

            packet = av.Packet(payload[offset:end])
            offset = end
            for frame in self.codec.decode(packet):
                image = frame.to_ndarray(format="bgr24")
                self._put_latest(image)

        if offset != len(payload):
            raise ProtocolError("unexpected bytes after video frame")

    def _put_latest(self, frame):
        while self.frames.full():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass

    def get_frame(self):
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def _clear_frames(self):
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return


class StreamReceiver:
    def __init__(self, host, port, decoder):
        self.host = host
        self.port = port
        self.decoder = decoder
        self.running = threading.Event()
        self.listen_socket = None
        self.connection = None
        self.thread = None
        self.error = None
        self.frames_received = 0

    def start(self):
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_socket.bind((self.host, self.port))
        self.listen_socket.listen(1)
        self.listen_socket.settimeout(0.5)
        self.running.set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Waiting for K230 on {self.host}:{self.port}")

    def _run(self):
        while self.running.is_set():
            try:
                connection, address = self.listen_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self.connection = connection
            connection.settimeout(5)
            print(f"K230 connected from {address[0]}:{address[1]}")
            try:
                self._receive_connection(connection)
            except (ConnectionError, OSError, ProtocolError) as error:
                if self.running.is_set():
                    print("Connection ended:", error)
            except Exception as error:
                if self.running.is_set():
                    print("Decode error; waiting for K230 to reconnect:", error)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass
                self.connection = None

    def _receive_connection(self, connection):
        config = recv_exact(connection, 12)
        data_type, width, height = struct.unpack("<LLL", config)
        if data_type != DATA_TYPE_CONFIG:
            raise ProtocolError("missing stream configuration")
        self.decoder.configure(width, height)

        while self.running.is_set():
            header = recv_exact(connection, 12)
            data_type, total_size, nalu_count = struct.unpack("<LLL", header)
            if data_type != DATA_TYPE_VIDEO_FRAME:
                raise ProtocolError(f"unknown data type 0x{data_type:08X}")
            if total_size > MAX_FRAME_BYTES:
                raise ProtocolError(f"frame is too large: {total_size} bytes")
            if not (0 < nalu_count <= MAX_NALU_COUNT):
                raise ProtocolError(f"invalid NALU count: {nalu_count}")

            payload = recv_exact(connection, total_size)
            self.decoder.decode_frame(payload, nalu_count)
            self.frames_received += 1

    def stop(self):
        self.running.clear()
        for sock in (self.connection, self.listen_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive and display a Yahboom K230 wireless H.265 stream"
    )
    parser.add_argument("--host", default="0.0.0.0", help="listen address")
    parser.add_argument("--port", type=int, default=6001, help="TCP port")
    return parser.parse_args()


def main():
    args = parse_args()
    decoder = H265Decoder()
    receiver = StreamReceiver(args.host, args.port, decoder)
    receiver.start()

    window_name = "Yahboom K230 Wireless Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_report = time.monotonic()
    last_count = 0

    try:
        while True:
            frame = decoder.get_frame()
            if frame is not None:
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            now = time.monotonic()
            if now - last_report >= 3:
                count = receiver.frames_received
                fps = (count - last_count) / (now - last_report)
                if count != last_count:
                    print(f"Receive rate: {fps:.1f} encoded frames/s")
                last_count = count
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
