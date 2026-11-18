"""Detect a reflective steel ball in a sensor-cropped grayscale image.

The sensor/ISP outputs only the narrow tube strip instead of a complete
640x480 frame. Detection combines a grayscale black-hat response, optional
circular refinement, and short temporal tracking without binary thresholding.
"""

import gc
import time

import cv2
import image
import launcher_control
import ulab.numpy as np
from machine import FPIOA, UART
from media.display import *
from media.media import *
from media.sensor import *


CAMERA_ID = 2
CAMERA_FPS = 60

# Use the 1280x720 source mode while preserving the old 1280-pixel-wide field
# of view. Crop a 1280x134 source strip and scale it to 640x67. The crop is
# centered on the previous 640x105 tube region.
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
SENSOR_CROP_X = 0
SENSOR_CROP_Y = 338
SENSOR_CROP_W = 1280
SENSOR_CROP_H = 134

FRAME_WIDTH = 640
FRAME_HEIGHT = 67
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Detection results are still reported in the original 640x480 coordinate
# system so calibration, logs, and UART output remain compatible.
REFERENCE_FRAME_WIDTH = 640
REFERENCE_FRAME_HEIGHT = 480
CAPTURE_ORIGIN_X = 0
CAPTURE_ORIGIN_Y = 229
CAMERA_CENTER_X = REFERENCE_FRAME_WIDTH // 2
CAMERA_CENTER_Y = REFERENCE_FRAME_HEIGHT // 2

# Horizontal scale calibration at the current fixed camera distance.
BALL_DIAMETER_MM = 10.0
BALL_DIAMETER_PIXELS = 20.0

# Detection uses the complete hardware-cropped frame. Only the search stage
# excludes a small margin at each side, so balls near the tube ends remain valid.
ROI_X = 0
ROI_Y = 0
ROI_W = FRAME_WIDTH
ROI_H = FRAME_HEIGHT

# Hough-circle parameters in unscaled ROI pixels.
HOUGH_DP = 1.0
HOUGH_MIN_DISTANCE = 18
HOUGH_CANNY_THRESHOLD = 60
HOUGH_SCORE_THRESHOLD = 12
MIN_RADIUS = 4
MAX_RADIUS = 30
GAUSSIAN_KERNEL = 3
GAUSSIAN_SIGMA = 0.8
ENABLE_HOUGH_REFINEMENT = False
HOUGH_EVERY_N_FRAMES = 5

# Grayscale dark-spot detector. It tolerates a bright reflection inside the
# ball because the surrounding dark rim still creates a strong response.
BLACKHAT_KERNEL_SIZE = 21
BLACKHAT_BLUR_KERNEL = 5
MIN_BLACKHAT_RESPONSE = 12.0
MIN_TRACKED_RESPONSE = 7.0
FALLBACK_RADIUS = 10
BLACKHAT_CENTER_Y_OFFSET = -5

# Exclude only 10 horizontal pixels on each side, which is less than the
# calibrated steel-ball diameter of about 20 pixels. Vertical limits are mapped
# from the old 105-row ROI into the centered 67-row hardware crop.
SEARCH_X_MARGIN = 10
SEARCH_Y_MIN = 8
SEARCH_Y_MAX = 60
MAX_ANCHOR_DISTANCE = 22
TRACK_SEARCH_X = 100
TRACK_SEARCH_Y = 28

# A steel ball should be darker on average than its local surroundings.
# Reflection may make part of it bright, so this threshold is intentionally low.
MIN_LOCAL_CONTRAST = 2.0
MAX_CIRCLES = 20
EDGE_MARGIN = 3

# Motion-aware tracking. Fast motion uses less smoothing and a short lead.
CONFIRM_FRAMES = 1
LOST_HOLD_FRAMES = 1
MAX_CENTER_JUMP = 160
POSITION_ALPHA = 0.70
FAST_POSITION_ALPHA = 0.92
FAST_MOTION_THRESHOLD = 6.0
RADIUS_ALPHA = 0.35
VELOCITY_ALPHA = 0.65
VELOCITY_DECAY = 0.75
DISPLAY_PREDICTION = 0.35
BOX_MARGIN = 3
AXIS_COLOR = (0, 0, 0)
DISPLAY_TO_IDE = False
DISPLAY_EVERY_N_FRAMES = 2
PRINT_EVERY_N_FRAMES = 60

# UART1: pin 9=TX, pin 10=RX.
UART_ENABLED = True
UART_TX_PIN = 9
UART_RX_PIN = 10
UART_BAUD = 115200
UART_SEND_EVERY_N_FRAMES = 1


def validate_config():
    if BALL_DIAMETER_MM <= 0 or BALL_DIAMETER_PIXELS <= 0:
        raise ValueError("ball diameter calibration must be positive")
    if (SENSOR_CROP_X < 0 or SENSOR_CROP_Y < 0 or
            SENSOR_CROP_W <= 0 or SENSOR_CROP_H <= 0):
        raise ValueError("invalid sensor crop")
    if (SENSOR_CROP_X + SENSOR_CROP_W > SENSOR_WIDTH or
            SENSOR_CROP_Y + SENSOR_CROP_H > SENSOR_HEIGHT):
        raise ValueError("sensor crop exceeds source mode")
    if (CAPTURE_ORIGIN_X < 0 or CAPTURE_ORIGIN_Y < 0 or
            CAPTURE_ORIGIN_X + FRAME_WIDTH > REFERENCE_FRAME_WIDTH or
            CAPTURE_ORIGIN_Y + FRAME_HEIGHT > REFERENCE_FRAME_HEIGHT):
        raise ValueError("capture mapping exceeds reference frame")
    if ROI_X < 0 or ROI_Y < 0 or ROI_W <= 0 or ROI_H <= 0:
        raise ValueError("invalid ROI")
    if ROI_X + ROI_W > FRAME_WIDTH or ROI_Y + ROI_H > FRAME_HEIGHT:
        raise ValueError("ROI exceeds the camera frame")
    if GAUSSIAN_KERNEL < 3 or GAUSSIAN_KERNEL % 2 == 0:
        raise ValueError("GAUSSIAN_KERNEL must be an odd number >= 3")
    if BLACKHAT_KERNEL_SIZE < 3 or BLACKHAT_KERNEL_SIZE % 2 == 0:
        raise ValueError("BLACKHAT_KERNEL_SIZE must be odd and >= 3")
    if BLACKHAT_BLUR_KERNEL < 3 or BLACKHAT_BLUR_KERNEL % 2 == 0:
        raise ValueError("BLACKHAT_BLUR_KERNEL must be odd and >= 3")
    if (SEARCH_X_MARGIN < 0 or SEARCH_X_MARGIN * 2 >= ROI_W or
            SEARCH_Y_MIN < 0 or SEARCH_Y_MAX > ROI_H or
            SEARCH_Y_MIN >= SEARCH_Y_MAX):
        raise ValueError("invalid grayscale search area")
    if HOUGH_EVERY_N_FRAMES < 1:
        raise ValueError("HOUGH_EVERY_N_FRAMES must be >= 1")
    if DISPLAY_EVERY_N_FRAMES < 1 or PRINT_EVERY_N_FRAMES < 1:
        raise ValueError("display/print interval must be >= 1")
    if UART_SEND_EVERY_N_FRAMES < 1:
        raise ValueError("UART_SEND_EVERY_N_FRAMES must be >= 1")


def pack_uart_frame(output):
    if output is None:
        return "N\r\n"
    # Signed horizontal distance in millimeters from tube/camera center x=320.
    return "%d\r\n" % output[7]


def patch_mean(gray, x0, y0, x1, y1):
    patch = gray[y0:y1, x0:x1]
    value = cv2.mean(patch)
    return float(value[0])


def local_contrast(gray, cx, cy, radius):
    """Return surrounding mean minus ball-region mean."""
    inner_r = max(3, radius)
    outer_r = radius + max(4, radius // 2)

    ix0 = max(0, cx - inner_r)
    iy0 = max(0, cy - inner_r)
    ix1 = min(ROI_W, cx + inner_r + 1)
    iy1 = min(ROI_H, cy + inner_r + 1)

    ox0 = max(0, cx - outer_r)
    oy0 = max(0, cy - outer_r)
    ox1 = min(ROI_W, cx + outer_r + 1)
    oy1 = min(ROI_H, cy + outer_r + 1)

    inner_mean = patch_mean(gray, ix0, iy0, ix1, iy1)
    outer_mean = patch_mean(gray, ox0, oy0, ox1, oy1)
    inner_area = (ix1 - ix0) * (iy1 - iy0)
    outer_area = (ox1 - ox0) * (oy1 - oy0)
    ring_area = outer_area - inner_area
    if ring_area <= 0:
        return 0.0

    ring_mean = (
        outer_mean * outer_area - inner_mean * inner_area
    ) / ring_area
    return ring_mean - inner_mean


def blackhat_peak(gray, blackhat_kernel, x0, y0, x1, y1, minimum):
    if x1 <= x0 or y1 <= y0:
        return None

    # Process only this search patch. During tracking this is much smaller
    # than the full tube ROI and is the main frame-rate optimization.
    patch = gray[y0:y1, x0:x1]
    blackhat = cv2.morphologyEx(
        patch, cv2.MORPH_BLACKHAT, blackhat_kernel
    )
    response = cv2.GaussianBlur(
        blackhat,
        (BLACKHAT_BLUR_KERNEL, BLACKHAT_BLUR_KERNEL),
        0,
    )
    guard = BLACKHAT_KERNEL_SIZE // 2
    patch_w = x1 - x0
    patch_h = y1 - y0
    if patch_w <= guard * 2 or patch_h <= guard * 2:
        return None

    # Morphology can create artificial peaks at a cropped patch boundary.
    # Ignore one kernel radius on all sides.
    valid_response = response[
        guard:patch_h - guard,
        guard:patch_w - guard,
    ]
    _, max_value, _, max_location = cv2.minMaxLoc(valid_response)
    if max_value < minimum:
        return None
    return (
        x0 + guard + int(max_location[0]),
        y0 + guard + int(max_location[1]),
        float(max_value),
    )


def detect_dark_spot(gray, blackhat_kernel, track):
    """Locate the moving dark feature, searching near prediction first."""
    peak = None
    if track["valid"]:
        predicted_x = int(track["cx"] + track["vx"] + 0.5)
        predicted_y = int(track["cy"] + track["vy"] + 0.5)
        x0 = max(SEARCH_X_MARGIN, predicted_x - TRACK_SEARCH_X)
        x1 = min(ROI_W - SEARCH_X_MARGIN, predicted_x + TRACK_SEARCH_X + 1)
        y0 = max(SEARCH_Y_MIN, predicted_y - TRACK_SEARCH_Y)
        y1 = min(SEARCH_Y_MAX, predicted_y + TRACK_SEARCH_Y + 1)
        peak = blackhat_peak(
            gray,
            blackhat_kernel,
            x0,
            y0,
            x1,
            y1,
            MIN_TRACKED_RESPONSE,
        )

    if peak is None:
        peak = blackhat_peak(
            gray,
            blackhat_kernel,
            SEARCH_X_MARGIN,
            SEARCH_Y_MIN,
            ROI_W - SEARCH_X_MARGIN,
            SEARCH_Y_MAX,
            MIN_BLACKHAT_RESPONSE,
        )
    if peak is None:
        return None

    cx = peak[0]
    cy = peak[1] + BLACKHAT_CENTER_Y_OFFSET
    cy = max(0, min(ROI_H - 1, cy))
    return (cx, cy, FALLBACK_RADIUS, peak[2])


def select_circle(gray, circles, anchor, last_track):
    """Refine a grayscale dark-spot anchor with a nearby Hough circle."""
    if anchor is None:
        return None
    if circles is None:
        return anchor

    best = None
    best_score = -1000000.0
    count = min(circles.shape[0], MAX_CIRCLES)

    for index in range(count):
        cx = int(circles[index, 0])
        cy = int(circles[index, 1])
        radius = int(circles[index, 2])

        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue
        if (cx - radius <= EDGE_MARGIN or cy - radius <= EDGE_MARGIN or
                cx + radius >= ROI_W - EDGE_MARGIN or
                cy + radius >= ROI_H - EDGE_MARGIN):
            continue

        anchor_distance = abs(cx - anchor[0]) + abs(cy - anchor[1])
        if anchor_distance > MAX_ANCHOR_DISTANCE:
            continue

        contrast = local_contrast(gray, cx, cy, radius)
        if contrast < MIN_LOCAL_CONTRAST:
            continue

        # The dark-spot anchor is primary; Hough only refines center and radius.
        score = contrast * 5.0 - anchor_distance
        if last_track["valid"]:
            distance = abs(cx - last_track["cx"]) + abs(cy - last_track["cy"])
            if distance <= MAX_CENTER_JUMP:
                score += MAX_CENTER_JUMP - distance
            else:
                score -= distance

        if score > best_score:
            best_score = score
            best = (cx, cy, radius, anchor[3])

    return best if best is not None else anchor


def update_track(track, candidate):
    if candidate is None:
        if track["valid"]:
            track["lost"] += 1
            if track["lost"] <= LOST_HOLD_FRAMES:
                # Bridge one blurred/missed frame using the measured velocity.
                track["cx"] += track["vx"]
                track["cy"] += track["vy"]
                track["vx"] *= VELOCITY_DECAY
                track["vy"] *= VELOCITY_DECAY
            else:
                track["valid"] = False
                track["hits"] = 0
        return

    cx, cy, radius, strength = candidate

    if track["valid"]:
        predicted_x = track["cx"] + track["vx"]
        predicted_y = track["cy"] + track["vy"]
        distance = abs(cx - predicted_x) + abs(cy - predicted_y)
        if distance <= MAX_CENTER_JUMP:
            measured_vx = cx - track["last_measure_x"]
            measured_vy = cy - track["last_measure_y"]
            velocity_alpha = VELOCITY_ALPHA
            track["vx"] = (
                track["vx"] * (1.0 - velocity_alpha) +
                measured_vx * velocity_alpha
            )
            track["vy"] = (
                track["vy"] * (1.0 - velocity_alpha) +
                measured_vy * velocity_alpha
            )

            speed = abs(measured_vx) + abs(measured_vy)
            position_alpha = (
                FAST_POSITION_ALPHA
                if speed >= FAST_MOTION_THRESHOLD
                else POSITION_ALPHA
            )
            track["cx"] += (cx - track["cx"]) * position_alpha
            track["cy"] += (cy - track["cy"]) * position_alpha
            track["radius"] += (
                radius - track["radius"]
            ) * RADIUS_ALPHA
            track["last_measure_x"] = float(cx)
            track["last_measure_y"] = float(cy)
            track["strength"] = strength
            track["hits"] += 1
            track["lost"] = 0
            return

    track["valid"] = True
    track["cx"] = float(cx)
    track["cy"] = float(cy)
    track["radius"] = float(radius)
    track["strength"] = strength
    track["vx"] = 0.0
    track["vy"] = 0.0
    track["last_measure_x"] = float(cx)
    track["last_measure_y"] = float(cy)
    track["hits"] = 1
    track["lost"] = 0


def current_output(track):
    """Return ROI position, frame position and error from camera center."""
    if not track["valid"] or track["hits"] < CONFIRM_FRAMES:
        return None

    cx = int(track["cx"] + track["vx"] * DISPLAY_PREDICTION + 0.5)
    cy = int(track["cy"] + track["vy"] * DISPLAY_PREDICTION + 0.5)
    cx = max(0, min(ROI_W - 1, cx))
    cy = max(0, min(ROI_H - 1, cy))
    radius = max(1, int(track["radius"] + 0.5))
    frame_x = CAPTURE_ORIGIN_X + ROI_X + cx
    frame_y = CAPTURE_ORIGIN_Y + ROI_Y + cy
    dx = frame_x - CAMERA_CENTER_X
    dy = frame_y - CAMERA_CENTER_Y
    dx_mm_float = dx * BALL_DIAMETER_MM / BALL_DIAMETER_PIXELS
    if dx_mm_float >= 0:
        dx_mm = int(dx_mm_float + 0.5)
    else:
        dx_mm = int(dx_mm_float - 0.5)
    return (cx, cy, radius, frame_x, frame_y, dx, dy, dx_mm)


def main():
    validate_config()

    sensor = None
    uart = None
    fpioa = None
    display_initialized = False
    media_initialized = False

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
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE),
    )

    track = {
        "valid": False,
        "cx": 0.0,
        "cy": 0.0,
        "radius": 0.0,
        "strength": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "last_measure_x": 0.0,
        "last_measure_y": 0.0,
        "hits": 0,
        "lost": 0,
    }

    try:
        sensor = Sensor(
            id=CAMERA_ID,
            width=SENSOR_WIDTH,
            height=SENSOR_HEIGHT,
            fps=CAMERA_FPS,
        )
        sensor.reset()
        sensor.set_framesize(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            chn=CAM_CHN_ID_0,
            crop=(
                SENSOR_CROP_X,
                SENSOR_CROP_Y,
                SENSOR_CROP_W,
                SENSOR_CROP_H,
            ),
        )
        sensor.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_0)

        Display.init(
            Display.ST7701,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            to_ide=DISPLAY_TO_IDE,
            quality=30,
        )
        display_initialized = True

        MediaManager.init()
        media_initialized = True

        if UART_ENABLED:
            fpioa = FPIOA()
            fpioa.set_function(UART_TX_PIN, FPIOA.UART1_TXD)
            fpioa.set_function(UART_RX_PIN, FPIOA.UART1_RXD)
            uart = UART(UART.UART1, UART_BAUD)

        sensor.run()
        time.sleep_ms(500)

        clock = time.clock()
        frame_count = 0
        print("grayscale ball tracker started")
        print("capture: %dx%d sensor_crop=(%d,%d,%d,%d)" % (
            FRAME_WIDTH,
            FRAME_HEIGHT,
            SENSOR_CROP_X,
            SENSOR_CROP_Y,
            SENSOR_CROP_W,
            SENSOR_CROP_H,
        ))
        print("ROI: x=%d y=%d w=%d h=%d reference_origin=(%d,%d)" % (
            ROI_X,
            ROI_Y,
            ROI_W,
            ROI_H,
            CAPTURE_ORIGIN_X,
            CAPTURE_ORIGIN_Y,
        ))
        if uart is not None:
            print("UART1: TX=%d RX=%d baud=%d center_x=%d" %
                  (UART_TX_PIN, UART_RX_PIN, UART_BAUD, CAMERA_CENTER_X))

        while True:
            launcher_control.checkpoint()
            clock.tick()

            frame = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_np = frame.to_numpy_ref()

            # The sensor output is already the complete 640x67 detector ROI.
            roi_gray = frame_np[
                ROI_Y:ROI_Y + ROI_H,
                ROI_X:ROI_X + ROI_W,
            ]
            anchor = detect_dark_spot(roi_gray, blackhat_kernel, track)
            candidate = anchor

            # Full-ROI Hough is expensive. Keep it optional and low-frequency;
            # grayscale dark-spot tracking supplies every-frame coordinates.
            if (ENABLE_HOUGH_REFINEMENT and anchor is not None and
                    frame_count % HOUGH_EVERY_N_FRAMES == 0):
                blurred = cv2.GaussianBlur(
                    roi_gray,
                    (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL),
                    GAUSSIAN_SIGMA,
                )
                circles = cv2.HoughCircles(
                    blurred,
                    3,
                    dp=HOUGH_DP,
                    minDist=HOUGH_MIN_DISTANCE,
                    param1=HOUGH_CANNY_THRESHOLD,
                    param2=HOUGH_SCORE_THRESHOLD,
                    minRadius=MIN_RADIUS,
                    maxRadius=MAX_RADIUS,
                )
                candidate = select_circle(blurred, circles, anchor, track)
            update_track(track, candidate)

            output = current_output(track)
            if output is not None:
                cx, cy, radius, center_x, center_y, dx, dy, dx_mm = output
                status = "BALL x:%dmm" % dx_mm
            else:
                cx = -1
                cy = -1
                radius = 0
                center_x = -1
                center_y = -1
                dx = 0
                dy = 0
                dx_mm = 0
                status = "SEARCHING" if track["valid"] else "NO BALL"

            if (uart is not None and
                    frame_count % UART_SEND_EVERY_N_FRAMES == 0):
                try:
                    uart.write(pack_uart_frame(output))
                except Exception as uart_error:
                    print("UART write error:", uart_error)

            # Display is intentionally decimated; detection and UART still run
            # on every captured frame.
            if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                preview = cv2.resize(roi_gray, (preview_w, preview_h))
                canvas_np[:, :] = 0
                canvas_np[
                    preview_y:preview_y + preview_h,
                    preview_x:preview_x + preview_w,
                ] = preview

                axis_roi_x = CAMERA_CENTER_X - ROI_X
                if 0 <= axis_roi_x < ROI_W:
                    axis_x = preview_x + int(axis_roi_x * preview_scale)
                    display_image.draw_line(
                        axis_x,
                        preview_y,
                        axis_x,
                        preview_y + preview_h - 1,
                        color=AXIS_COLOR,
                        thickness=1,
                    )

                if output is not None:
                    x0 = max(0, cx - radius - BOX_MARGIN)
                    y0 = max(0, cy - radius - BOX_MARGIN)
                    x1 = min(ROI_W, cx + radius + BOX_MARGIN)
                    y1 = min(ROI_H, cy + radius + BOX_MARGIN)

                    draw_x = preview_x + int(x0 * preview_scale)
                    draw_y = preview_y + int(y0 * preview_scale)
                    draw_w = max(1, int((x1 - x0) * preview_scale))
                    draw_h = max(1, int((y1 - y0) * preview_scale))
                    display_image.draw_rectangle(
                        draw_x,
                        draw_y,
                        draw_w,
                        draw_h,
                        color=(0, 0, 0),
                        thickness=2,
                    )

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
                    "ball=(%d,%d) x=%dmm err_px=(%d,%d) v=(%.1f,%.1f) resp=%.1f fps=%.1f" %
                    (
                        center_x,
                        center_y,
                        dx_mm,
                        dx,
                        dy,
                        track["vx"],
                        track["vy"],
                        track["strength"],
                        clock.fps(),
                    )
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
        if uart is not None:
            try:
                uart.deinit()
            except Exception:
                pass
        if display_initialized:
            Display.deinit()
        launcher_control.checkpoint(0)
        time.sleep_ms(100)
        if media_initialized:
            MediaManager.deinit()


if __name__ == "__main__":
    main()
