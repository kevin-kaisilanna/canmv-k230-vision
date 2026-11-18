# 分组图像采集 - 红黄蓝 × 矩形/三角/圆
import time
import os
from media.sensor import *
from media.display import *
from media.media import *
import image
import ybUtils.YbKey as YbKey

# ------------------------ 配置参数 ------------------------
SAVE_BASE = "/data/data/image/"          # 存储根目录
GROUPS = [
    "red_rect", "yellow_rect", "blue_rect",
    "red_tri", "yellow_tri", "blue_tri",
    "red_circle", "yellow_circle", "blue_circle"
]
PHOTO_COUNT = 100                        # 每组拍摄数量
INTERVAL_MS = 50                         # 拍摄间隔(毫秒)

# ------------------------ 工具函数 ------------------------
def ensure_dir(directory):
    """递归创建目录"""
    if not directory or directory == '/':
        return
    directory = directory.rstrip('/')
    try:
        os.stat(directory)
        return
    except OSError:
        pass

    # 递归创建父目录
    if '/' in directory:
        parent = directory[:directory.rindex('/')]
        if parent and parent != directory:
            ensure_dir(parent)
    try:
        os.mkdir(directory)
        print(f"目录已创建: {directory}")
    except OSError:
        pass

# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    try:
        # 初始化按键
        key = YbKey.YbKey()

        # 初始化摄像头
        sensor = Sensor()
        sensor.reset()
        sensor.set_framesize(width=640, height=480, chn=CAM_CHN_ID_1)
        sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)

        # 初始化显示屏（用于实时预览和提示）
        Display.init(Display.ST7701, width=640, height=480, to_ide=True)
        MediaManager.init()
        sensor.run()

        # 状态变量
        current_group_index = 0           # 当前组索引 (0~8)
        last_key_status = False           # 按键上一次状态
        collecting = False                # 是否正在采集

        print("分组采集程序已启动，按下 Key 按钮开始采集")
        while True:
            # 捕获图像用于预览
            img = sensor.snapshot(chn=CAM_CHN_ID_1)
            # 创建一个副本用于绘制提示信息
            disp_img = image.Image(640, 480, image.RGB565)
            disp_img.clear()
            disp_img.copy_from(img)

            # 显示当前组信息和状态
            info_line1 = "Group: {}/{} - {}".format(
                current_group_index + 1, len(GROUPS), GROUPS[current_group_index])
            info_line2 = "Press Key to start" if not collecting else "...Collecting..."
            disp_img.draw_string_advanced(10, 10, 25, info_line1, color=(255,255,255))
            disp_img.draw_string_advanced(10, 50, 25, info_line2, color=(0,255,0))
            Display.show_image(disp_img, 0, 0, Display.LAYER_OSD2)

            # 检测按键（防抖处理：只在按下瞬间触发）
            current_key = (key.is_pressed() == 1)
            if current_key and not last_key_status:
                if not collecting:
                    # 开始采集当前组
                    group_name = GROUPS[current_group_index]
                    group_dir = SAVE_BASE + group_name + "/"
                    ensure_dir(group_dir)

                    print("开始采集: {} (共{}张，间隔{}ms)".format(group_name, PHOTO_COUNT, INTERVAL_MS))
                    for i in range(1, PHOTO_COUNT + 1):
                        # 拍摄
                        shot = sensor.snapshot(chn=CAM_CHN_ID_1)
                        # 保存图片（文件名如：001.jpg）
                        filename = "{:03d}.jpg".format(i)
                        filepath = group_dir + filename
                        shot.save(filepath)
                        print("已保存: {}".format(filepath))
                        # 更新预览提示
                        disp_img2 = image.Image(640, 480, image.RGB565)
                        disp_img2.clear()
                        disp_img2.copy_from(shot)
                        disp_img2.draw_string_advanced(10, 10, 25,
                            "Collecting {} - {}/{}".format(group_name, i, PHOTO_COUNT),
                            color=(255,255,0))
                        Display.show_image(disp_img2, 0, 0, Display.LAYER_OSD2)
                        # 间隔
                        if i < PHOTO_COUNT:
                            time.sleep_ms(INTERVAL_MS)

                    print("采集完成: {}".format(group_name))
                    # 切换到下一组（循环）
                    current_group_index = (current_group_index + 1) % len(GROUPS)
                    collecting = False
            # 更新按键历史状态
            last_key_status = current_key

            # 释放CPU，避免占用过高
            time.sleep_ms(10)

    except KeyboardInterrupt:
        print("用户停止程序")
    except BaseException as e:
        print(f"异常: {e}")
    finally:
        if isinstance(sensor, Sensor):
            sensor.stop()
        Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        MediaManager.deinit()
