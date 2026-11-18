"""High-speed steel-ball tracker based on a one-dimensional contrast scanline.

This implementation follows the recognition idea described in the 2026 H-task
analysis document:

1. Mount the camera above the horizontal tube.
2. Use the sensor crop window to acquire only the 640x67 tube region.
3. Process only the horizontal row crossing the ball center.
4. Compare grayscale pixels separated by two pixels with difference-over-sum.
5. Select the strongest contrast change instead of using a fixed threshold.

For a usable ball-center coordinate, the default mode pairs the strongest
bright-to-dark edge with a nearby dark-to-bright edge. If a valid edge pair is
not available, it falls back to the single strongest contrast point and shifts
it by half of the calibrated ball diameter. Detection itself uses no binary
threshold, contours, Canny, or Hough circles.

This file is independent from grayscale_ball_tracker.py and can be copied to
the K230 together with launcher_control.py, then run directly.
"""

import gc
import time

import image
import launcher_control
import ulab.numpy as np
from machine import FPIOA, UART
from media.display import *
from media.media import *
from media.sensor import *


CAMERA_ID = 2
CAMERA_FPS = 60

# Preserve the old 1280-pixel-wide field of view so the tube and ball keep the
# same apparent size as the original 1280x960 version. The 1280x720 source mode
# is used as a faster compromise, and the old y=420 crop becomes approximately
# y=300 after the mode's centered vertical crop.
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
SENSOR_CROP_X = 0
SENSOR_CROP_Y = 300
SENSOR_CROP_W = 1280
SENSOR_CROP_H = 134

FRAME_WIDTH = 640
FRAME_HEIGHT = 67
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
CAMERA_CENTER_X = FRAME_WIDTH // 2
CAMERA_CENTER_Y = FRAME_HEIGHT // 2

# Horizontal scale calibration at the current fixed camera distance.
BALL_DIAMETER_MM = 10.0
BALL_DIAMETER_PIXELS = 20.0

# The sensor output is already the tube region, so detection uses the whole
# 640x67 frame without an additional NumPy crop or copy.
ROI_X = 0
ROI_Y = 0
ROI_W = FRAME_WIDTH
ROI_H = FRAME_HEIGHT

# One-dimensional scanline configuration. PIXEL_GAP=2 implements the document's
# "two-pixel interval" comparison: pixel[x] is compared with pixel[x + 2].
SCANLINE_Y = ROI_H // 2
SEARCH_X_MIN = 30
SEARCH_X_MAX = ROI_W - 30
PIXEL_GAP = 2
CONTRAST_SCALE = 256

# Pairing two opposite edges gives the actual center rather than returning one
# side of the ball. Disable this to use only the single strongest edge fallback.
USE_EDGE_PAIR = True
EDGE_CANDIDATE_COUNT = 8
MIN_BALL_WIDTH = 12
MAX_BALL_WIDTH = 32
EXPECTED_BALL_WIDTH = 20
WIDTH_PENALTY = 2
TRACK_PENALTY_DIVISOR = 2
DARKNESS_BONUS_DIVISOR = 2
BACKGROUND_SAMPLE_WIDTH = 5

# One-dimensional temporal tracking. The multi-frame velocity follows the
# document's suggestion of comparing the current position with 10 frames ago.
POSITION_ALPHA = 0.72
FAST_POSITION_ALPHA = 0.94
FAST_MOTION_THRESHOLD = 6.0
VELOCITY_HISTORY_FRAMES = 10
VELOCITY_ALPHA = 0.60
DISPLAY_PREDICTION = 0.35
MAX_REACQUIRE_JUMP = 180

# Keep the control loop independent from the slow IDE image-transfer path.
# ST7701 uses a persistent full-screen buffer for stable DMA display. Only the
# 67-row camera strip is copied into it; no cv2 resize is performed.
DISPLAY_TO_IDE = False
DISPLAY_EVERY_N_FRAMES = 1
DISPLAY_IMAGE_X = (DISPLAY_WIDTH - FRAME_WIDTH) // 2
DISPLAY_IMAGE_Y = (DISPLAY_HEIGHT - FRAME_HEIGHT) // 2
DISPLAY_STATUS_TEXT = False
PRINT_EVERY_N_FRAMES = 60
BOX_HALF_HEIGHT = 16
AXIS_COLOR = (0, 0, 0)

# UART1: pin 9=TX, pin 10=RX. The output protocol matches the existing tracker:
# signed horizontal distance in millimeters, or "N" when unavailable.
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
        raise ValueError("sensor crop exceeds sensor input")
    if ROI_X < 0 or ROI_Y < 0 or ROI_W <= 0 or ROI_H <= 0:
        raise ValueError("invalid ROI")
    if ROI_X + ROI_W > FRAME_WIDTH or ROI_Y + ROI_H > FRAME_HEIGHT:
        raise ValueError("ROI exceeds camera frame")
    if SCANLINE_Y < 0 or SCANLINE_Y >= ROI_H:
        raise ValueError("SCANLINE_Y is outside ROI")
    if PIXEL_GAP < 1:
        raise ValueError("PIXEL_GAP must be >= 1")
    if (SEARCH_X_MIN < 0 or SEARCH_X_MAX > ROI_W or
            SEARCH_X_MAX - SEARCH_X_MIN <= PIXEL_GAP):
        raise ValueError("invalid scanline search range")
    if EDGE_CANDIDATE_COUNT < 2:
        raise ValueError("EDGE_CANDIDATE_COUNT must be >= 2")
    if MIN_BALL_WIDTH < PIXEL_GAP or MAX_BALL_WIDTH < MIN_BALL_WIDTH:
        raise ValueError("invalid ball width range")
    if EXPECTED_BALL_WIDTH < MIN_BALL_WIDTH or EXPECTED_BALL_WIDTH > MAX_BALL_WIDTH:
        raise ValueError("EXPECTED_BALL_WIDTH is outside width range")
    if VELOCITY_HISTORY_FRAMES < 1:
        raise ValueError("VELOCITY_HISTORY_FRAMES must be >= 1")
    if DISPLAY_EVERY_N_FRAMES < 1 or PRINT_EVERY_N_FRAMES < 1:
        raise ValueError("display/print interval must be >= 1")
    if UART_SEND_EVERY_N_FRAMES < 1:
        raise ValueError("UART_SEND_EVERY_N_FRAMES must be >= 1")


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def rounded_int(value):
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


def interval_mean(prefix, x0, x1):
    """Return a scanline interval mean using the precomputed prefix sum."""
    x0 = clamp(x0, 0, ROI_W)
    x1 = clamp(x1, 0, ROI_W)
    if x1 <= x0:
        return 0
    return (prefix[x1] - prefix[x0]) // (x1 - x0)


def insert_edge_candidate(xs, strengths, x, strength):
    """Insert one edge into a fixed-size strongest-first candidate list."""
    last = EDGE_CANDIDATE_COUNT - 1
    if strength <= strengths[last]:
        return

    index = last
    while index > 0 and strength > strengths[index - 1]:
        strengths[index] = strengths[index - 1]
        xs[index] = xs[index - 1]
        index -= 1
    strengths[index] = strength
    xs[index] = x


def build_contrast_profile(
        row,
        profile,
        prefix,
        falling_xs,
        falling_strengths,
        rising_xs,
        rising_strengths):
    """Build the scanline profile and retain only its strongest edge candidates.

    Pairing every falling edge with every nearby rising edge is expensive in
    MicroPython. Keeping the strongest few edges changes pairing from thousands
    of Python iterations per frame to at most EDGE_CANDIDATE_COUNT squared.
    """
    prefix[0] = 0
    running_sum = 0
    for x in range(ROI_W):
        running_sum += int(row[x])
        prefix[x + 1] = running_sum

    for index in range(EDGE_CANDIDATE_COUNT):
        falling_xs[index] = -1
        falling_strengths[index] = 0
        rising_xs[index] = -1
        rising_strengths[index] = 0

    end = SEARCH_X_MAX - PIXEL_GAP
    strongest_x = SEARCH_X_MIN
    strongest_value = 0
    strongest_abs = -1

    for x in range(SEARCH_X_MIN, end):
        left_value = int(row[x])
        right_value = int(row[x + PIXEL_GAP])
        denominator = left_value + right_value + 1
        value = ((right_value - left_value) * CONTRAST_SCALE) // denominator
        profile[x] = value

        if value < 0:
            insert_edge_candidate(
                falling_xs, falling_strengths, x, -value
            )
        elif value > 0:
            insert_edge_candidate(
                rising_xs, rising_strengths, x, value
            )

        magnitude = abs(value)
        if magnitude > strongest_abs:
            strongest_abs = magnitude
            strongest_x = x
            strongest_value = value

    return strongest_x, strongest_value, strongest_abs


def pair_ball_edges(
        prefix,
        predicted_x,
        falling_xs,
        falling_strengths,
        rising_xs,
        rising_strengths):
    """Pair the strongest bright-dark and dark-bright edge candidates."""
    best_score = -1000000
    best_left = -1
    best_right = -1
    best_strength = 0

    for left_index in range(EDGE_CANDIDATE_COUNT):
        left_x = falling_xs[left_index]
        if left_x < 0:
            continue
        falling_strength = falling_strengths[left_index]

        for right_index in range(EDGE_CANDIDATE_COUNT):
            right_x = rising_xs[right_index]
            if right_x < 0:
                continue
            width = right_x - left_x
            if width < MIN_BALL_WIDTH or width > MAX_BALL_WIDTH:
                continue
            rising_strength = rising_strengths[right_index]
            center = (left_x + right_x + PIXEL_GAP) // 2
            score = falling_strength + rising_strength
            score -= abs(width - EXPECTED_BALL_WIDTH) * WIDTH_PENALTY

            inner_mean = interval_mean(
                prefix,
                left_x + PIXEL_GAP,
                right_x + PIXEL_GAP,
            )
            left_bg = interval_mean(
                prefix,
                left_x - BACKGROUND_SAMPLE_WIDTH,
                left_x,
            )
            right_bg = interval_mean(
                prefix,
                right_x + PIXEL_GAP,
                right_x + PIXEL_GAP + BACKGROUND_SAMPLE_WIDTH,
            )
            background_mean = (left_bg + right_bg) // 2
            darkness = background_mean - inner_mean
            if darkness > 0:
                score += darkness // DARKNESS_BONUS_DIVISOR

            if predicted_x is not None:
                score -= abs(center - predicted_x) // TRACK_PENALTY_DIVISOR

            if score > best_score:
                best_score = score
                best_left = left_x + PIXEL_GAP // 2
                best_right = right_x + PIXEL_GAP // 2
                best_strength = falling_strength + rising_strength

    if best_left < 0:
        return None

    center = (best_left + best_right) // 2
    return (center, best_left, best_right, best_strength, best_score, True)


def strongest_edge_fallback(strongest_x, strongest_value, strongest_abs):
    """Estimate the center from the globally strongest contrast point."""
    edge_x = strongest_x + PIXEL_GAP // 2
    half_width = EXPECTED_BALL_WIDTH // 2

    if strongest_value < 0:
        # Bright-to-dark: treat it as the left side of the ball.
        center = edge_x + half_width
        left_edge = edge_x
        right_edge = edge_x + EXPECTED_BALL_WIDTH
    else:
        # Dark-to-bright: treat it as the right side of the ball.
        center = edge_x - half_width
        left_edge = edge_x - EXPECTED_BALL_WIDTH
        right_edge = edge_x

    center = clamp(center, SEARCH_X_MIN, SEARCH_X_MAX - 1)
    left_edge = clamp(left_edge, SEARCH_X_MIN, SEARCH_X_MAX - 1)
    right_edge = clamp(right_edge, SEARCH_X_MIN, SEARCH_X_MAX - 1)
    return (center, left_edge, right_edge, strongest_abs, strongest_abs, False)


def detect_ball_on_scanline(
        row,
        profile,
        prefix,
        track,
        falling_xs,
        falling_strengths,
        rising_xs,
        rising_strengths):
    strongest_x, strongest_value, strongest_abs = build_contrast_profile(
        row,
        profile,
        prefix,
        falling_xs,
        falling_strengths,
        rising_xs,
        rising_strengths,
    )

    predicted_x = None
    if track["valid"]:
        predicted_x = rounded_int(track["cx"] + track["vx"])
        predicted_x = clamp(predicted_x, SEARCH_X_MIN, SEARCH_X_MAX - 1)

    candidate = None
    if USE_EDGE_PAIR:
        candidate = pair_ball_edges(
            prefix,
            predicted_x,
            falling_xs,
            falling_strengths,
            rising_xs,
            rising_strengths,
        )
    if candidate is None:
        candidate = strongest_edge_fallback(
            strongest_x,
            strongest_value,
            strongest_abs,
        )
    return candidate


def reset_track(track, candidate):
    center, left_edge, right_edge, strength, score, paired = candidate
    track["valid"] = True
    track["cx"] = float(center)
    track["vx"] = 0.0
    track["last_measure"] = float(center)
    track["left"] = int(left_edge)
    track["right"] = int(right_edge)
    track["strength"] = int(strength)
    track["score"] = int(score)
    track["paired"] = paired
    track["history"] = [float(center)]


def update_track(track, candidate):
    if candidate is None:
        track["valid"] = False
        track["history"] = []
        return

    center, left_edge, right_edge, strength, score, paired = candidate
    if not track["valid"]:
        reset_track(track, candidate)
        return

    predicted = track["cx"] + track["vx"]
    if abs(center - predicted) > MAX_REACQUIRE_JUMP:
        reset_track(track, candidate)
        return

    measured_step = center - track["last_measure"]
    position_alpha = (
        FAST_POSITION_ALPHA
        if abs(measured_step) >= FAST_MOTION_THRESHOLD
        else POSITION_ALPHA
    )
    track["cx"] += (center - track["cx"]) * position_alpha
    track["last_measure"] = float(center)

    history = track["history"]
    history.append(float(center))
    if len(history) > VELOCITY_HISTORY_FRAMES:
        old_position = history.pop(0)
        interval = VELOCITY_HISTORY_FRAMES
        measured_velocity = (center - old_position) / interval
        track["vx"] = (
            track["vx"] * (1.0 - VELOCITY_ALPHA) +
            measured_velocity * VELOCITY_ALPHA
        )

    track["left"] = int(left_edge)
    track["right"] = int(right_edge)
    track["strength"] = int(strength)
    track["score"] = int(score)
    track["paired"] = paired


def current_output(track):
    if not track["valid"]:
        return None

    roi_x = rounded_int(track["cx"] + track["vx"] * DISPLAY_PREDICTION)
    roi_x = clamp(roi_x, 0, ROI_W - 1)
    frame_x = ROI_X + roi_x
    frame_y = ROI_Y + SCANLINE_Y
    dx = frame_x - CAMERA_CENTER_X
    dy = frame_y - CAMERA_CENTER_Y
    dx_mm = rounded_int(dx * BALL_DIAMETER_MM / BALL_DIAMETER_PIXELS)

    return (
        roi_x,
        SCANLINE_Y,
        frame_x,
        frame_y,
        dx,
        dy,
        dx_mm,
        track["left"],
        track["right"],
        track["strength"],
        track["score"],
        track["vx"],
        track["paired"],
    )


def pack_uart_frame(output):
    if output is None:
        return "N\r\n"
    return "%d\r\n" % output[6]


def main():
    validate_config()

    sensor = None
    uart = None
    display_initialized = False
    media_initialized = False

    # Use a persistent display-sized buffer. Directly sending the odd-height
    # sensor frame can produce corrupted ST7701 output because of layer/DMA
    # alignment and sensor-buffer reuse.
    canvas_np = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=np.uint8)
    display_image = image.Image(
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        image.GRAYSCALE,
        alloc=image.ALLOC_REF,
        data=canvas_np,
    )

    # Reused buffers avoid allocating a complete contrast list every frame.
    profile = [0] * ROI_W
    prefix = [0] * (ROI_W + 1)
    falling_xs = [-1] * EDGE_CANDIDATE_COUNT
    falling_strengths = [0] * EDGE_CANDIDATE_COUNT
    rising_xs = [-1] * EDGE_CANDIDATE_COUNT
    rising_strengths = [0] * EDGE_CANDIDATE_COUNT
    track = {
        "valid": False,
        "cx": 0.0,
        "vx": 0.0,
        "last_measure": 0.0,
        "left": 0,
        "right": 0,
        "strength": 0,
        "score": 0,
        "paired": False,
        "history": [],
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
        print("scanline contrast ball tracker started")
        print("capture: %dx%d sensor_crop=(%d,%d,%d,%d)" % (
            FRAME_WIDTH,
            FRAME_HEIGHT,
            SENSOR_CROP_X,
            SENSOR_CROP_Y,
            SENSOR_CROP_W,
            SENSOR_CROP_H,
        ))
        print("ROI: x=%d y=%d w=%d h=%d scan_y=%d gap=%d" % (
            ROI_X,
            ROI_Y,
            ROI_W,
            ROI_H,
            SCANLINE_Y,
            PIXEL_GAP,
        ))
        if uart is not None:
            print("UART1: TX=%d RX=%d baud=%d center_x=%d" % (
                UART_TX_PIN,
                UART_RX_PIN,
                UART_BAUD,
                CAMERA_CENTER_X,
            ))

        while True:
            launcher_control.checkpoint()
            clock.tick()

            frame = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_np = frame.to_numpy_ref()
            roi_gray = frame_np

            row = roi_gray[SCANLINE_Y]
            candidate = detect_ball_on_scanline(
                row,
                profile,
                prefix,
                track,
                falling_xs,
                falling_strengths,
                rising_xs,
                rising_strengths,
            )
            update_track(track, candidate)
            output = current_output(track)

            if uart is not None and frame_count % UART_SEND_EVERY_N_FRAMES == 0:
                try:
                    uart.write(pack_uart_frame(output))
                except Exception as uart_error:
                    print("UART write error:", uart_error)

            if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                # Copy only the native 640x67 strip into the persistent LCD
                # canvas. The copy also decouples LCD DMA from sensor buffers.
                canvas_np[
                    DISPLAY_IMAGE_Y:DISPLAY_IMAGE_Y + FRAME_HEIGHT,
                    DISPLAY_IMAGE_X:DISPLAY_IMAGE_X + FRAME_WIDTH,
                ] = roi_gray

                axis_roi_x = CAMERA_CENTER_X - ROI_X
                if 0 <= axis_roi_x < ROI_W:
                    axis_x_draw = DISPLAY_IMAGE_X + axis_roi_x
                    display_image.draw_line(
                        axis_x_draw,
                        DISPLAY_IMAGE_Y,
                        axis_x_draw,
                        DISPLAY_IMAGE_Y + FRAME_HEIGHT - 1,
                        color=AXIS_COLOR,
                        thickness=1,
                    )

                if output is not None:
                    roi_x = output[0]
                    left_edge = output[7]
                    right_edge = output[8]
                    draw_left = DISPLAY_IMAGE_X + left_edge
                    draw_right = DISPLAY_IMAGE_X + right_edge
                    draw_center = DISPLAY_IMAGE_X + roi_x
                    draw_top = DISPLAY_IMAGE_Y + clamp(
                        SCANLINE_Y - BOX_HALF_HEIGHT,
                        0,
                        FRAME_HEIGHT - 1,
                    )
                    draw_bottom = DISPLAY_IMAGE_Y + clamp(
                        SCANLINE_Y + BOX_HALF_HEIGHT,
                        0,
                        FRAME_HEIGHT - 1,
                    )

                    display_image.draw_rectangle(
                        draw_left,
                        draw_top,
                        max(1, draw_right - draw_left),
                        max(1, draw_bottom - draw_top),
                        color=(0, 0, 0),
                        thickness=2,
                    )
                    display_image.draw_line(
                        draw_center,
                        draw_top,
                        draw_center,
                        draw_bottom,
                        color=(0, 0, 0),
                        thickness=2,
                    )
                    if DISPLAY_STATUS_TEXT:
                        mode = "PAIR" if output[12] else "EDGE"
                        status = "%s x:%dmm c:%d" % (
                            mode,
                            output[6],
                            output[9],
                        )
                        display_image.draw_string_advanced(
                            8,
                            8,
                            18,
                            status,
                            color=(255, 255, 255),
                        )
                elif DISPLAY_STATUS_TEXT:
                    display_image.draw_string_advanced(
                        8,
                        8,
                        18,
                        "NO BALL",
                        color=(255, 255, 255),
                    )

                Display.show_image(display_image)

            frame_count += 1
            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                if output is not None:
                    print(
                        "ball=(%d,%d) x=%dmm edges=(%d,%d) "
                        "strength=%d score=%d v=%.2f mode=%s fps=%.1f" % (
                            output[2],
                            output[3],
                            output[6],
                            output[7],
                            output[8],
                            output[9],
                            output[10],
                            output[11],
                            "pair" if output[12] else "edge",
                            clock.fps(),
                        )
                    )
                else:
                    print("no ball fps=%.1f" % clock.fps())
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
