import time
import lvgl as lv
from media.display import *
from media.media import *
from machine import TOUCH

# 导入识别函数
from ver import run_rect_recognition
from cls_video import classification

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# LVGL 触摸驱动
class touch_screen():
    def __init__(self):
        self.state = lv.INDEV_STATE.RELEASED
        self.indev_drv = lv.indev_create()
        self.indev_drv.set_type(lv.INDEV_TYPE.POINTER)
        self.indev_drv.set_read_cb(self.callback)
        self.touch = TOUCH(0)

    def callback(self, driver, data):
        x, y, state = 0, 0, lv.INDEV_STATE.RELEASED
        tp = self.touch.read(1)
        if tp:
            x, y, event = tp[0].x, tp[0].y, tp[0].event
            if event == 2 or event == 3:
                state = lv.INDEV_STATE.PRESSED
        data.point = lv.point_t({'x': x, 'y': y})
        data.state = state

# 显示刷新回调
disp_img1 = None
def disp_drv_flush_cb(disp_drv, area, color):
    global disp_img1
    Display.show_image(disp_img1)
    time.sleep(0.01)
    disp_drv.flush_ready()

# 初始化显示和 LVGL
def display_init():
    Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
    MediaManager.init()

def lvgl_init():
    global disp_img1
    lv.init()
    disp_img1 = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.BGRA8888)
    disp_drv = lv.disp_create(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    disp_drv.set_draw_buffers(disp_img1.bytearray(), None, disp_img1.size(), lv.DISP_RENDER_MODE.DIRECT)
    disp_drv.set_flush_cb(disp_drv_flush_cb)
    touch_screen()

def display_deinit():
    Display.deinit()
    MediaManager.deinit()

def lvgl_deinit():
    global disp_img1
    lv.deinit()
    del disp_img1
    disp_img1 = None

# 按钮回调
def on_rect_click(event):
    if event.code == lv.EVENT.CLICKED:
        # 卸载 LVGL 和显示
        lvgl_deinit()
        display_deinit()
        # 运行矩形识别（内部会自己初始化硬件）
        run_rect_recognition()
        # 返回后重新初始化 LVGL 主菜单
        display_init()
        lvgl_init()
        create_main_menu()

def on_shape_click(event):
    if event.code == lv.EVENT.CLICKED:
        lvgl_deinit()
        display_deinit()
        classification()
        display_init()
        lvgl_init()
        create_main_menu()

def create_main_menu():
    lv.obj_clean(lv.scr_act())   # 清空屏幕
    btn1 = lv.btn(lv.scr_act())
    btn1.set_size(200, 60)
    btn1.align(lv.ALIGN.CENTER, -120, 0)
    lv.label(btn1).set_text("矩形识别")
    btn1.add_event_cb(on_rect_click, lv.EVENT.CLICKED, None)

    btn2 = lv.btn(lv.scr_act())
    btn2.set_size(200, 60)
    btn2.align(lv.ALIGN.CENTER, 120, 0)
    lv.label(btn2).set_text("图形检测")
    btn2.add_event_cb(on_shape_click, lv.EVENT.CLICKED, None)

# 主入口
if __name__ == "__main__":
    try:
        display_init()
        lvgl_init()
        create_main_menu()
        while True:
            period = lv.timer_handler_run_in_period(5)
            time.sleep_ms(period)
    except KeyboardInterrupt:
        pass
    finally:
        lvgl_deinit()
        display_deinit()
