"""Two-course launcher for Yahboom K230.

Touch button "1" starts grayscale_ball_tracker.
Touch button "2" starts rtsp_wireless_stream.
Press the Yahboom Key used by cam.py to return to this menu.
"""

import gc
import os
import select
import sys
import time

import image
from machine import TOUCH
from media.display import Display
from media.media import MediaManager

import ybUtils.YbKey as YbKey

import grayscale_ball_tracker
import launcher_control
import rtsp_wireless_stream


DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
BUTTON_1 = (70, 120, 220, 220)
BUTTON_2 = (350, 120, 220, 220)
MENU_REFRESH_MS = 50


def point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def draw_button(canvas, rect, label):
    x, y, w, h = rect
    canvas.draw_rectangle(x, y, w, h, color=(255, 255, 255), thickness=4)
    canvas.draw_string_advanced(
        x + w // 2 - 24,
        y + h // 2 - 48,
        96,
        label,
        color=(255, 255, 255),
    )


def draw_menu(canvas, message="Touch 1 or 2"):
    canvas.clear()
    canvas.draw_string_advanced(48, 34, 32, "K230 Course Launcher",
                                color=(255, 255, 255))
    canvas.draw_string_advanced(60, 78, 24, message, color=(0, 255, 0))
    draw_button(canvas, BUTTON_1, "1")
    draw_button(canvas, BUTTON_2, "2")
    canvas.draw_string_advanced(66, 360, 22, "1: grayscale ball tracker",
                                color=(255, 255, 255))
    canvas.draw_string_advanced(66, 396, 22, "2: RTSP wireless stream",
                                color=(255, 255, 255))
    canvas.draw_string_advanced(66, 432, 20, "In a course, press Key to return",
                                color=(255, 255, 0))
    Display.show_image(canvas)


def read_touch_choice(tp, was_touching):
    points = tp.read()
    touching = len(points) > 0
    choice = None

    if touching and not was_touching:
        point = points[0]
        try:
            x = int(point.x)
            y = int(point.y)
        except AttributeError:
            x = int(point[0])
            y = int(point[1])
        if point_in_rect(x, y, BUTTON_1):
            choice = "1"
        elif point_in_rect(x, y, BUTTON_2):
            choice = "2"

    return choice, touching


def read_console_choice():
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1)
    except Exception:
        pass
    return None


class ReturnKeyGuard:
    def __init__(self):
        self.key = YbKey.YbKey()
        self.last_pressed = False

    def should_return(self):
        pressed = False
        try:
            pressed = self.key.is_pressed() == 1
        except Exception:
            return False

        edge = pressed and not self.last_pressed
        self.last_pressed = pressed
        return edge


def run_course(module, name):
    """Run a blocking course main() and let Yahboom Key raise KeyboardInterrupt."""
    guard = ReturnKeyGuard()
    launcher_control.set_return_checker(guard.should_return)
    try:
        print("[launcher] starting %s" % name)
        module.main()
    finally:
        launcher_control.set_return_checker(None)
        gc.collect()
        print("[launcher] returned to menu")


def init_menu():
    Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT,
                 to_ide=True)
    MediaManager.init()
    return TOUCH(0), image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.RGB565)


def deinit_menu():
    try:
        Display.deinit()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    gc.collect()


def main():
    courses = {
        "1": (grayscale_ball_tracker, "grayscale_ball_tracker"),
        "2": (rtsp_wireless_stream, "rtsp_wireless_stream"),
    }

    while True:
        tp = None
        canvas = None
        was_touching = False

        try:
            tp, canvas = init_menu()
            draw_menu(canvas)

            while True:
                launcher_control.checkpoint()
                choice, was_touching = read_touch_choice(tp, was_touching)
                if choice is None:
                    key = read_console_choice()
                    if key in courses:
                        choice = key

                if choice in courses:
                    module, name = courses[choice]
                    deinit_menu()
                    run_course(module, name)
                    break

                time.sleep_ms(MENU_REFRESH_MS)

        except KeyboardInterrupt:
            print("[launcher] stopped by user")
            break
        except BaseException as error:
            print("[launcher] error:", error)
            try:
                if canvas is not None:
                    draw_menu(canvas, "Error, see IDE terminal")
                    time.sleep_ms(1000)
            except Exception:
                pass
        finally:
            deinit_menu()


if __name__ == "__main__":
    main()
