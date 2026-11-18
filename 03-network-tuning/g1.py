import time, os, select, struct
from media.sensor import *
from media.display import *
from media.media import *
from machine import Timer
from cl import PC_Discoverer
from g1lib import StreamSender
from vofa_lite import JustFloatHandler
import config

if __name__ == "__main__":
    try:
        # ============================================================
        # 网络配置 - 请在 config.py 中修改 IP 和网络模式
        # ============================================================
        # PC_Discoverer 是针对 PC 自动发现的类，但为了稳定，
        # 在此版本仅使用连接功能。详情可查阅 cl.py
        discoverer = PC_Discoverer(
            is_wlan=config.USE_WLAN,
            ssid=config.WIFI_SSID,
            password=config.WIFI_PASSWORD,
            ip=config.K230_IP,
            mask=config.K230_MASK,
            gate=config.K230_GATEWAY,
            dns=config.K230_DNS,
            udp_port=config.UDP_PORT,
            timeout=config.CONNECT_TIMEOUT
        )

        # PC 端 IP（请确保 config.py 中的 PC_IP 正确）
        ip = config.PC_IP

        # 初始化传感器
        sensor = Sensor(id=2, width=1280, height=720, fps=90)
        sensor.reset()

        # 创建处理器实例并传入用户回调
        def user_callback(floats):
            for ch, value in enumerate(floats):
                print(f"  通道{ch+1}: {value:.6f}")
        handler = JustFloatHandler(user_callback)

        # 创建视频流发送器
        sender = StreamSender(
            sensor=sensor,
            cam_chn_id=CAM_CHN_ID_1,
            enc_chn_id=VENC_CHN_ID_0,
            width=1280,
            height=720,
            host_ip=ip,
            port=config.TCP_PORT,
            message_callback=handler.handle_socket
        )

        # 设置其他通道（非流媒体通道）
        sensor.set_framesize(chn=CAM_CHN_ID_2, width=640, height=360)
        sensor.set_pixformat(chn=CAM_CHN_ID_2, pix_format=Sensor.YUV420SP)
        sensor.set_framesize(chn=CAM_CHN_ID_0, width=640, height=360)
        sensor.set_pixformat(chn=CAM_CHN_ID_0, pix_format=Sensor.RGB565)

        # 显示初始化
        Display.init(Display.ST7701, width=800, height=480, osd_num=2, to_ide=False)
        # 注释掉下面函数即可自由使用 CAM_CHN_ID_2
        Display.bind_layer(src=(6, 0, CAM_CHN_ID_2), dstlayer=Display.LAYER_VIDEO1,
                          rect=(480-360, 0, 640, 360), pix_format=Sensor.YUV420SP)

        MediaManager.init()
        sender.start()
        sensor.run()

        clock = time.clock()
        # 创建定时器并传入用户回调
        tim = Timer(-1)
        def send_mseeage_callback(unuse):
            sender.send_message(f"time={time.time()}, fps={clock.fps()}")
            print("[定时器回调]:已发送数据")
        tim.init(period=3000, mode=Timer.PERIODIC, callback=send_mseeage_callback)

        timel = time.ticks_us()
        while True:
            clock.tick()
            timel = time.ticks_us()

            sender.send_frame()     # 发送视频帧
            sender.poll_messages()  # 检查并处理消息
            if sender.isClose:
                print("pc主动断开连接")
                break

            tu = time.ticks_us() - timel
            if tu < 11111:
                if tu < 0:
                    time.sleep_us(11111)
                else:
                    time.sleep_us(11111 - tu)


    except Exception as e:
        print("主函数错误:", e)
    finally:
        try:
            if 'sender' in locals():
                sender.close()
        except:
            pass
        if 'sensor' in locals() and isinstance(sensor, Sensor):
            sensor.stop()
        Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        MediaManager.deinit()
