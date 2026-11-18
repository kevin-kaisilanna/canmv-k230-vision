"""按键触发拍照：一次保存 50 张图片到检测目录。"""

import os
import time

import image
from media.display import *
from media.media import *
from media.sensor import *
import ybUtils.YbKey as YbKey


SAVE_DIR = "/data/data/image/jiance"
PHOTO_COUNT = 20
INTERVAL_MS = 200
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def ensure_dir(directory):
    """递归创建目录，目录已经存在时不报错。"""
    if not directory or directory == "/":
        return

    directory = directory.rstrip("/")
    try:
        os.stat(directory)
        return
    except OSError:
        pass

    separator = directory.rfind("/")
    if separator > 0:
        ensure_dir(directory[:separator])

    try:
        os.mkdir(directory)
        print("目录已创建:", directory)
    except OSError:
        # 可能由另一个任务在检查后创建，再确认一次。
        os.stat(directory)


def next_available_index(directory, start=1):
    """返回第一个不存在的 NNN.jpg 编号，避免覆盖已有图片。"""
    index = start
    while True:
        path = "%s/%03d.jpg" % (directory, index)
        try:
            os.stat(path)
            index += 1
        except OSError:
            return index


def show_preview(display_image, frame, status, color=(0, 255, 0)):
    display_image.clear()
    display_image.copy_from(frame)
    display_image.draw_string_advanced(
        10,
        10,
        26,
        status,
        color=color,
    )
    Display.show_image(display_image, 0, 0, Display.LAYER_OSD2)


def main():
    sensor = None
    media_initialized = False

    try:
        ensure_dir(SAVE_DIR)
        next_index = next_available_index(SAVE_DIR)

        key = YbKey.YbKey()

        sensor = Sensor()
        sensor.reset()
        sensor.set_framesize(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            chn=CAM_CHN_ID_1,
        )
        sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)

        Display.init(
            Display.ST7701,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            to_ide=True,
        )
        MediaManager.init()
        media_initialized = True
        sensor.run()

        display_image = image.Image(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            image.RGB565,
        )

        last_key_status = False
        capture_done = False
        print("按下 Key，程序将保存 %d 张图片到: %s" %
              (PHOTO_COUNT, SAVE_DIR))

        while True:
            frame = sensor.snapshot(chn=CAM_CHN_ID_1)

            if capture_done:
                status = "Saved %d images" % PHOTO_COUNT
                status_color = (0, 255, 0)
            else:
                status = "Press Key: save %d images" % PHOTO_COUNT
                status_color = (255, 255, 0)
            show_preview(display_image, frame, status, status_color)

            current_key = (key.is_pressed() == 1)
            if current_key and not last_key_status and not capture_done:
                print("检测到按键，开始采集")

                for count in range(1, PHOTO_COUNT + 1):
                    shot = sensor.snapshot(chn=CAM_CHN_ID_1)
                    filepath = "%s/%03d.jpg" % (SAVE_DIR, next_index)
                    shot.save(filepath)
                    print("已保存 %d/%d: %s" %
                          (count, PHOTO_COUNT, filepath))

                    show_preview(
                        display_image,
                        shot,
                        "Saving %d/%d" % (count, PHOTO_COUNT),
                        (255, 255, 0),
                    )
                    next_index += 1

                    if count < PHOTO_COUNT:
                        time.sleep_ms(INTERVAL_MS)

                capture_done = True
                print("采集完成，共保存 %d 张图片" % PHOTO_COUNT)

            last_key_status = current_key
            time.sleep_ms(10)

    except KeyboardInterrupt:
        print("用户停止程序")
    except BaseException as error:
        print("异常:", error)
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        if media_initialized:
            MediaManager.deinit()


if __name__ == "__main__":
    main()
