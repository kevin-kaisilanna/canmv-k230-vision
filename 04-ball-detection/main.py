"""Crop the tube ROI, binarize it, and locate one dark steel ball.

Target: Yahboom CanMV K230 with the current OpenCV-enabled firmware.
Only the cropped ROI is processed. The display shows a binary image where
the tube is white and the steel ball is black.
"""

import gc
import os
import time

import cv2
import image
import ulab.numpy as np
from media.display import *
from media.media import *
from media.sensor import *


# Camera and display configuration.
CAMERA_ID = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Tube ROI in the original 640x480 camera image.
# Adjust these four values first if the tube is not fully inside the crop.
ROI_X = 20
ROI_Y = 210
ROI_W = 600
ROI_H = 105

# Pixels <= DARK_THRESHOLD become black in the displayed binary image.
# Raise it if the ball is incomplete; lower it if tube shadows are detected.
DARK_THRESHOLD = 150

# Candidate filters in the unscaled ROI coordinate system.
MIN_AREA = 25
MAX_AREA = 3000
MIN_SIZE = 5
MAX_SIZE = 75
MIN_ASPECT = 0.50
MAX_ASPECT = 1.80
MIN_FILL_RATIO = 0.32
EDGE_MARGIN = 3

MORPH_KERNEL_SIZE = 3
BOX_MARGIN = 3
PRINT_EVERY_N_FRAMES = 5


def validate_config():
    if ROI_X < 0 or ROI_Y < 0 or ROI_W <= 0 or ROI_H <= 0:
        raise ValueError("invalid ROI")
    if ROI_X + ROI_W > FRAME_WIDTH or ROI_Y + ROI_H > FRAME_HEIGHT:
        raise ValueError("ROI exceeds the camera frame")


def find_ball(dark_mask):
    """Return the best ball candidate as (x, y, w, h, area), or None."""
    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
            continue

        if (x <= EDGE_MARGIN or y <= EDGE_MARGIN or
                x + w >= ROI_W - EDGE_MARGIN or
                y + h >= ROI_H - EDGE_MARGIN):
            continue

        aspect = float(w) / h
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < MIN_FILL_RATIO:
            continue

        # Prefer a large, compact and nearly round dark region.
        roundness = 1.0 - abs(w - h) / float(max(w, h))
        score = area * fill_ratio * (0.5 + 0.5 * roundness)
        if score > best_score:
            best_score = score
            best = (x, y, w, h, area)

    return best


def clamp_box(x, y, w, h):
    x0 = max(0, x - BOX_MARGIN)
    y0 = max(0, y - BOX_MARGIN)
    x1 = min(ROI_W, x + w + BOX_MARGIN)
    y1 = min(ROI_H, y + h + BOX_MARGIN)
    return x0, y0, x1 - x0, y1 - y0


def main():
    validate_config()

    sensor = None
    display_initialized = False
    media_initialized = False

    # Keep the ROI aspect ratio in the full-screen preview.
    preview_scale = min(
        DISPLAY_WIDTH / float(ROI_W),
        DISPLAY_HEIGHT / float(ROI_H),
    )
    preview_w = int(ROI_W * preview_scale)
    preview_h = int(ROI_H * preview_scale)
    preview_x = (DISPLAY_WIDTH - preview_w) // 2
    preview_y = (DISPLAY_HEIGHT - preview_h) // 2

    canvas_np = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=np.uint8)
    display_image = image.Image(
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        image.GRAYSCALE,
        alloc=image.ALLOC_REF,
        data=canvas_np,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )

    try:
        sensor = Sensor(
            id=CAMERA_ID,
            width=1280,
            height=960,
            fps=30,
        )
        sensor.reset()
        sensor.set_framesize(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            chn=CAM_CHN_ID_0,
        )
        sensor.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_0)

        Display.init(
            Display.ST7701,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            to_ide=True,
            quality=50,
        )
        display_initialized = True

        MediaManager.init()
        media_initialized = True
        sensor.run()
        time.sleep_ms(500)

        clock = time.clock()
        frame_count = 0
        print("binary ball detector started")
        print("ROI: x=%d y=%d w=%d h=%d threshold=%d" %
              (ROI_X, ROI_Y, ROI_W, ROI_H, DARK_THRESHOLD))

        while True:
            os.exitpoint()
            clock.tick()

            frame = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_np = frame.to_numpy_ref()

            # Crop first. All expensive processing below runs only on this ROI.
            roi_gray = frame_np[
                ROI_Y:ROI_Y + ROI_H,
                ROI_X:ROI_X + ROI_W,
            ]
            blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)

            # Display polarity: white tube, black steel ball.
            _, binary = cv2.threshold(
                blurred,
                DARK_THRESHOLD,
                255,
                cv2.THRESH_BINARY,
            )

            # Contour polarity: dark objects become white for findContours().
            dark_mask = cv2.bitwise_not(binary)
            dark_mask = cv2.morphologyEx(
                dark_mask, cv2.MORPH_OPEN, kernel
            )
            dark_mask = cv2.morphologyEx(
                dark_mask, cv2.MORPH_CLOSE, kernel
            )
            ball = find_ball(dark_mask)

            preview = cv2.resize(binary, (preview_w, preview_h))
            canvas_np[:, :] = 0
            canvas_np[
                preview_y:preview_y + preview_h,
                preview_x:preview_x + preview_w,
            ] = preview

            if ball is not None:
                x, y, w, h, area = ball
                bx, by, bw, bh = clamp_box(x, y, w, h)

                draw_x = preview_x + int(bx * preview_scale)
                draw_y = preview_y + int(by * preview_scale)
                draw_w = max(1, int(bw * preview_scale))
                draw_h = max(1, int(bh * preview_scale))
                display_image.draw_rectangle(
                    draw_x,
                    draw_y,
                    draw_w,
                    draw_h,
                    color=(0, 0, 0),
                    thickness=2,
                )

                center_x = ROI_X + x + w // 2
                center_y = ROI_Y + y + h // 2
                status = "BALL x:%d y:%d" % (center_x, center_y)
            else:
                area = 0
                center_x = -1
                center_y = -1
                status = "NO BALL"

            display_image.draw_string_advanced(
                8,
                8,
                24,
                status,
                color=(255, 255, 255),
            )
            Display.show_image(display_image)

            frame_count += 1
            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                print(
                    "ball=(%d,%d) area=%d fps=%.1f" %
                    (center_x, center_y, int(area), clock.fps())
                )
                gc.collect()

    except KeyboardInterrupt:
        print("user stopped")
    except BaseException as error:
        print("error:", error)
        raise
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        if display_initialized:
            Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        if media_initialized:
            MediaManager.deinit()


if __name__ == "__main__":
    main()
