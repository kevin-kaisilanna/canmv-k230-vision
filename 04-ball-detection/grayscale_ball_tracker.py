"""Detect a reflective steel ball in a tightly cropped grayscale tube strip.

This is independent from main.py, so the original binary detector is unchanged.
Detection uses a bright temporal background, compact-shape validation, and
motion-aware tracking to suppress stationary tube borders and seams.
"""

import gc
import os
import time

import cv2
import image
import ulab.numpy as np
from machine import FPIOA, UART
from media.display import *
from media.media import *
from media.sensor import *


CAMERA_ID = 2
CAMERA_FPS = 90
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
CAMERA_CENTER_X = FRAME_WIDTH // 2
CAMERA_CENTER_Y = FRAME_HEIGHT // 2

# Horizontal scale calibration at the current fixed camera distance.
BALL_DIAMETER_MM = 10.0
BALL_DIAMETER_PIXELS = 20.0

# First-stage ROI used by the previous version.
SOURCE_ROI_X = 0
SOURCE_ROI_Y = 210
SOURCE_ROI_W = 640
SOURCE_ROI_H = 105

# Second-stage crop measured from the captured tube view. The strip is moved
# upward by 5 pixels and enlarged so the ball is not cut off at the bottom.
# Detection retains the complete horizontal travel in a 640x67 strip.
TUBE_CROP_X = 0
TUBE_CROP_Y = 38
TUBE_CROP_W = 640
TUBE_CROP_H = 67

# Final detection ROI in the original 640x480 camera coordinates.
ROI_X = SOURCE_ROI_X + TUBE_CROP_X
ROI_Y = SOURCE_ROI_Y + TUBE_CROP_Y
ROI_W = TUBE_CROP_W
ROI_H = TUBE_CROP_H

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

# Geometric validation for black-hat response contours. Tube borders and
# seams are long/thin; the steel-ball response must be compact and roughly
# circular, while still allowing some horizontal motion blur.
MIN_RESPONSE_AREA = 8
MAX_RESPONSE_AREA = 800
MIN_RESPONSE_SIZE = 3
MAX_RESPONSE_WIDTH = 40
MAX_RESPONSE_HEIGHT = 36
MIN_RESPONSE_ASPECT = 0.30
MAX_RESPONSE_ASPECT = 3.00
EDGE_MIN_RESPONSE_ASPECT = 0.20
EDGE_MAX_RESPONSE_ASPECT = 4.00
MIN_RESPONSE_FILL = 0.10
MIN_CANDIDATE_CONTRAST = 1.0

# Primary detector: compare the current tube strip with a slowly leaking
# per-pixel bright reference. The stationary tube borders and center seam
# cancel out; a steel ball remains as a compact darker-than-background area.
BACKGROUND_WARMUP_FRAMES = 30
BACKGROUND_UPDATE_EVERY_N_FRAMES = 6
BACKGROUND_DECAY = 1
MIN_BACKGROUND_RESPONSE = 10.0
MIN_TRACKED_BACKGROUND_RESPONSE = 6.0
BACKGROUND_THRESHOLD_RATIO = 0.42
MIN_BACKGROUND_AREA = 12
MAX_BACKGROUND_AREA = 900
MIN_BACKGROUND_SIZE = 4
MAX_BACKGROUND_WIDTH = 44
MAX_BACKGROUND_HEIGHT = 42
MIN_BACKGROUND_ASPECT = 0.30
MAX_BACKGROUND_ASPECT = 3.20
MIN_BACKGROUND_FILL = 0.12
BACKGROUND_BLUR_KERNEL = 3
BACKGROUND_OPEN_KERNEL = 3
BACKGROUND_CLOSE_KERNEL = 5
BACKGROUND_FREEZE_MARGIN = 10

# The ball rolls close to this horizontal path in the tightly cropped strip.
# This rejects the tube's upper/lower borders without limiting x at all.
BALL_PATH_Y = 52
BALL_PATH_TOLERANCE = 17
TRACK_PATH_TOLERANCE = 21

# Search the complete horizontal camera width, including both tube ends.
SEARCH_X_MARGIN = 0
SEARCH_Y_MIN = 0
SEARCH_Y_MAX = ROI_H
MAX_ANCHOR_DISTANCE = 22
TRACK_SEARCH_X = 100
TRACK_SEARCH_Y = 20

# A steel ball should be darker on average than its local surroundings.
# Reflection may make part of it bright, so this threshold is intentionally low.
MIN_LOCAL_CONTRAST = 2.0
MAX_CIRCLES = 20
EDGE_MARGIN = 3

# Motion-aware tracking. Fast motion uses less smoothing and a short lead.
CONFIRM_FRAMES = 3
LOST_HOLD_FRAMES = 4
MAX_CENTER_JUMP = 100
POSITION_ALPHA = 0.70
FAST_POSITION_ALPHA = 0.92
FAST_MOTION_THRESHOLD = 6.0
RADIUS_ALPHA = 0.35
VELOCITY_ALPHA = 0.65
VELOCITY_DECAY = 0.75
DISPLAY_PREDICTION = 0.35
BOX_MARGIN = 3
AXIS_COLOR = (0, 0, 0)
DISPLAY_EVERY_N_FRAMES = 2
PRINT_EVERY_N_FRAMES = 30

# UART1: pin 9=TX, pin 10=RX.
UART_ENABLED = True
UART_TX_PIN = 9
UART_RX_PIN = 10
UART_BAUD = 115200
UART_SEND_EVERY_N_FRAMES = 1


def validate_config():
    if BALL_DIAMETER_MM <= 0 or BALL_DIAMETER_PIXELS <= 0:
        raise ValueError("ball diameter calibration must be positive")
    if (SOURCE_ROI_X < 0 or SOURCE_ROI_Y < 0 or
            SOURCE_ROI_W <= 0 or SOURCE_ROI_H <= 0):
        raise ValueError("invalid source ROI")
    if (SOURCE_ROI_X + SOURCE_ROI_W > FRAME_WIDTH or
            SOURCE_ROI_Y + SOURCE_ROI_H > FRAME_HEIGHT):
        raise ValueError("source ROI exceeds the camera frame")
    if (TUBE_CROP_X < 0 or TUBE_CROP_Y < 0 or
            TUBE_CROP_W <= 0 or TUBE_CROP_H <= 0):
        raise ValueError("invalid tube crop")
    if (TUBE_CROP_X + TUBE_CROP_W > SOURCE_ROI_W or
            TUBE_CROP_Y + TUBE_CROP_H > SOURCE_ROI_H):
        raise ValueError("tube crop exceeds the source ROI")
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
    if (BACKGROUND_BLUR_KERNEL < 3 or
            BACKGROUND_BLUR_KERNEL % 2 == 0):
        raise ValueError("BACKGROUND_BLUR_KERNEL must be odd and >= 3")
    if (BACKGROUND_OPEN_KERNEL < 3 or BACKGROUND_OPEN_KERNEL % 2 == 0 or
            BACKGROUND_CLOSE_KERNEL < 3 or
            BACKGROUND_CLOSE_KERNEL % 2 == 0):
        raise ValueError("background morphology kernels must be odd and >= 3")
    if (BACKGROUND_WARMUP_FRAMES < 2 or
            BACKGROUND_UPDATE_EVERY_N_FRAMES < 1 or
            BACKGROUND_DECAY < 1):
        raise ValueError("invalid background update parameters")
    if (BACKGROUND_THRESHOLD_RATIO <= 0.0 or
            BACKGROUND_THRESHOLD_RATIO > 1.0):
        raise ValueError("BACKGROUND_THRESHOLD_RATIO must be in (0, 1]")
    if (BALL_PATH_Y < 0 or BALL_PATH_Y >= ROI_H or
            BALL_PATH_TOLERANCE < 1 or TRACK_PATH_TOLERANCE < 1):
        raise ValueError("invalid ball path parameters")
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


def background_peak(gray, background, open_kernel, close_kernel, track):
    """Return the best compact region that became darker than the tube."""
    response = cv2.subtract(background, gray)
    response = cv2.GaussianBlur(
        response,
        (BACKGROUND_BLUR_KERNEL, BACKGROUND_BLUR_KERNEL),
        0,
    )

    _, maximum, _, _ = cv2.minMaxLoc(response)
    minimum = (
        MIN_TRACKED_BACKGROUND_RESPONSE
        if track["valid"] else MIN_BACKGROUND_RESPONSE
    )
    if maximum < minimum:
        return None

    threshold_value = max(
        int(minimum),
        int(float(maximum) * BACKGROUND_THRESHOLD_RATIO),
    )
    _, response_mask = cv2.threshold(
        response, threshold_value, 255, cv2.THRESH_BINARY
    )
    response_mask = cv2.morphologyEx(
        response_mask, cv2.MORPH_OPEN, open_kernel
    )
    response_mask = cv2.morphologyEx(
        response_mask, cv2.MORPH_CLOSE, close_kernel
    )
    contours, _ = cv2.findContours(
        response_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if track["valid"]:
        predicted_x = track["cx"] + track["vx"]
        predicted_y = track["cy"] + track["vy"]
        path_tolerance = TRACK_PATH_TOLERANCE
    else:
        predicted_x = 0.0
        predicted_y = float(BALL_PATH_Y)
        path_tolerance = BALL_PATH_TOLERANCE

    best = None
    best_score = -1000000.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_BACKGROUND_AREA or area > MAX_BACKGROUND_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if (bw < MIN_BACKGROUND_SIZE or bh < MIN_BACKGROUND_SIZE or
                bw > MAX_BACKGROUND_WIDTH or
                bh > MAX_BACKGROUND_HEIGHT):
            continue

        # Reject long tube borders/seams. A ball at x=0 or x=639 is allowed.
        if bh >= ROI_H - 4 or bw >= ROI_W // 8:
            continue

        aspect = float(bw) / bh
        if (aspect < MIN_BACKGROUND_ASPECT or
                aspect > MAX_BACKGROUND_ASPECT):
            continue

        fill_ratio = area / float(bw * bh)
        if fill_ratio < MIN_BACKGROUND_FILL:
            continue

        cx = bx + bw // 2
        cy = by + bh // 2
        path_error = abs(cy - BALL_PATH_Y)
        if path_error > path_tolerance:
            continue

        response_patch = response[by:by + bh, bx:bx + bw]
        mean_response = float(cv2.mean(response_patch)[0])
        _, strength, _, peak_location = cv2.minMaxLoc(response_patch)
        peak_x = bx + int(peak_location[0])

        # Keep x responsive by blending the stable box center with the strongest
        # dark response. This also reduces jitter from the ball's reflection.
        cx = int((cx * 3 + peak_x) / 4)
        contrast = local_contrast(gray, cx, cy, max(4, min(bw, bh) // 2))
        if contrast < MIN_CANDIDATE_CONTRAST:
            continue

        tracking_bonus = 0.0
        if track["valid"]:
            distance = abs(cx - predicted_x) + abs(cy - predicted_y)
            if distance > MAX_CENTER_JUMP:
                continue
            tracking_bonus = (MAX_CENTER_JUMP - distance) * 2.5

        roundness = 1.0 - abs(bw - bh) / float(max(bw, bh))
        score = (
            float(strength) * 2.5 +
            mean_response * 3.0 +
            area * (0.20 + fill_ratio) +
            roundness * 35.0 +
            contrast * 6.0 +
            tracking_bonus -
            path_error * 2.0
        )
        if score > best_score:
            best_score = score
            radius = max(4, min(MAX_RADIUS, int((bw + bh) / 4)))
            best = (cx, cy, radius, float(strength))

    return best


def update_bright_background(background, gray, decay_image, track):
    """Update a leaky bright reference while protecting the tracked ball."""
    frozen = None
    fx0 = 0
    fy0 = 0
    fx1 = 0
    fy1 = 0
    if track["valid"]:
        freeze_radius = int(track["radius"] + BACKGROUND_FREEZE_MARGIN)
        fx0 = max(0, int(track["cx"]) - freeze_radius)
        fy0 = max(0, int(track["cy"]) - freeze_radius)
        fx1 = min(ROI_W, int(track["cx"]) + freeze_radius + 1)
        fy1 = min(ROI_H, int(track["cy"]) + freeze_radius + 1)
        frozen = background[fy0:fy1, fx0:fx1].copy()

    # A one-level decay lets the reference follow slow illumination changes.
    # Taking the brighter value immediately removes old ball positions.
    decayed = cv2.subtract(background, decay_image)
    rise = cv2.subtract(gray, decayed)
    updated = cv2.add(decayed, rise)

    if frozen is not None:
        updated[fy0:fy1, fx0:fx1] = frozen
    return updated


def blackhat_peak(gray, blackhat_kernel, x0, y0, x1, y1, minimum):
    if x1 <= x0 or y1 <= y0:
        return None

    # Black-hat creates candidate regions; geometry and local contrast decide
    # whether a region can actually be the steel ball.
    patch = gray[y0:y1, x0:x1]
    blackhat = cv2.morphologyEx(
        patch, cv2.MORPH_BLACKHAT, blackhat_kernel
    )
    response = cv2.GaussianBlur(
        blackhat,
        (BLACKHAT_BLUR_KERNEL, BLACKHAT_BLUR_KERNEL),
        0,
    )
    _, response_mask = cv2.threshold(
        response, int(minimum), 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        response_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    guard = BLACKHAT_KERNEL_SIZE // 2
    patch_w = x1 - x0
    patch_h = y1 - y0

    # Local search-window borders are artificial and must be ignored. The real
    # x=0/x=639 camera boundaries remain usable, but the upper/lower tube
    # borders are always excluded by one kernel radius.
    left_limit = guard if x0 > 0 else 0
    right_limit = patch_w - guard if x1 < ROI_W else patch_w
    top_limit = guard
    bottom_limit = patch_h - guard
    if right_limit <= left_limit or bottom_limit <= top_limit:
        return None

    best = None
    best_score = -1000000.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_RESPONSE_AREA or area > MAX_RESPONSE_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if (bw < MIN_RESPONSE_SIZE or bh < MIN_RESPONSE_SIZE or
                bw > MAX_RESPONSE_WIDTH or
                bh > MAX_RESPONSE_HEIGHT):
            continue

        candidate_x = bx + bw // 2
        candidate_y = by + bh // 2
        if (candidate_x < left_limit or candidate_x >= right_limit or
                candidate_y < top_limit or candidate_y >= bottom_limit):
            continue

        # A candidate touching a real left/right image boundary may be a
        # partially visible ball, so its aspect allowance is slightly wider.
        frame_left = x0 + bx
        frame_right = frame_left + bw
        touches_x_edge = frame_left <= 1 or frame_right >= ROI_W - 1
        min_aspect = (
            EDGE_MIN_RESPONSE_ASPECT
            if touches_x_edge else MIN_RESPONSE_ASPECT
        )
        max_aspect = (
            EDGE_MAX_RESPONSE_ASPECT
            if touches_x_edge else MAX_RESPONSE_ASPECT
        )
        aspect = float(bw) / bh
        if aspect < min_aspect or aspect > max_aspect:
            continue

        fill_ratio = area / float(bw * bh)
        if fill_ratio < MIN_RESPONSE_FILL:
            continue

        contour_response = response[by:by + bh, bx:bx + bw]
        _, strength, _, peak_location = cv2.minMaxLoc(contour_response)
        peak_x = bx + int(peak_location[0])
        peak_y = by + int(peak_location[1])
        center_x = x0 + peak_x
        center_y = y0 + peak_y + BLACKHAT_CENTER_Y_OFFSET
        center_y = max(0, min(ROI_H - 1, center_y))

        contrast = local_contrast(
            gray, center_x, center_y, FALLBACK_RADIUS
        )
        if contrast < MIN_CANDIDATE_CONTRAST:
            continue

        roundness = 1.0 - abs(bw - bh) / float(max(bw, bh))
        score = (
            float(strength) * 2.0 +
            area * (0.25 + fill_ratio) +
            roundness * 40.0 +
            contrast * 5.0
        )
        if score > best_score:
            best_score = score
            best = (center_x, y0 + peak_y, float(strength))

    return best

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
    frame_x = ROI_X + cx
    frame_y = ROI_Y + cy
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
    background_open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BACKGROUND_OPEN_KERNEL, BACKGROUND_OPEN_KERNEL),
    )
    background_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BACKGROUND_CLOSE_KERNEL, BACKGROUND_CLOSE_KERNEL),
    )
    # Fill after allocation: ulab may promote uint8 when an array is
    # multiplied by a Python integer, which OpenCV rejects in mixed-type ops.
    background_decay_image = np.zeros(
        (ROI_H, ROI_W), dtype=np.uint8
    )
    background_decay_image[:, :] = BACKGROUND_DECAY
    background = None
    background_warmup = 0

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
            width=1280,
            height=960,
            fps=CAMERA_FPS,
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
        print("tube ROI: x=%d y=%d w=%d h=%d" %
              (ROI_X, ROI_Y, ROI_W, ROI_H))
        print("calibration: keep tube empty or roll ball for %d frames" %
              BACKGROUND_WARMUP_FRAMES)
        if uart is not None:
            print("UART1: TX=%d RX=%d baud=%d center_x=%d" %
                  (UART_TX_PIN, UART_RX_PIN, UART_BAUD, CAMERA_CENTER_X))

        while True:
            os.exitpoint()
            clock.tick()

            frame = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_np = frame.to_numpy_ref()

            # Crop twice: first to the old detection band, then tightly to the
            # white tube. All following processing only sees the final strip.
            source_gray = frame_np[
                SOURCE_ROI_Y:SOURCE_ROI_Y + SOURCE_ROI_H,
                SOURCE_ROI_X:SOURCE_ROI_X + SOURCE_ROI_W,
            ]
            roi_gray = source_gray[
                TUBE_CROP_Y:TUBE_CROP_Y + TUBE_CROP_H,
                TUBE_CROP_X:TUBE_CROP_X + TUBE_CROP_W,
            ]
            detector_gray = cv2.GaussianBlur(
                roi_gray,
                (BACKGROUND_BLUR_KERNEL, BACKGROUND_BLUR_KERNEL),
                0,
            )

            calibrating = False
            if background is None:
                # Allocate the reference with an explicit OpenCV-compatible
                # uint8 type instead of inheriting an ulab temporary type.
                background = np.zeros(
                    (ROI_H, ROI_W), dtype=np.uint8
                )
                background[:, :] = detector_gray
                background_warmup = 1
                candidate = None
                calibrating = True
            elif background_warmup < BACKGROUND_WARMUP_FRAMES:
                rise = cv2.subtract(detector_gray, background)
                background = cv2.add(background, rise)
                background_warmup += 1
                candidate = None
                calibrating = True
            else:
                candidate = background_peak(
                    detector_gray,
                    background,
                    background_open_kernel,
                    background_close_kernel,
                    track,
                )

            anchor = candidate
            if (ENABLE_HOUGH_REFINEMENT and anchor is not None and
                    frame_count % HOUGH_EVERY_N_FRAMES == 0):
                circles = cv2.HoughCircles(
                    detector_gray,
                    3,
                    dp=HOUGH_DP,
                    minDist=HOUGH_MIN_DISTANCE,
                    param1=HOUGH_CANNY_THRESHOLD,
                    param2=HOUGH_SCORE_THRESHOLD,
                    minRadius=MIN_RADIUS,
                    maxRadius=MAX_RADIUS,
                )
                candidate = select_circle(
                    detector_gray, circles, anchor, track
                )

            update_track(track, candidate)
            if (not calibrating and
                    frame_count % BACKGROUND_UPDATE_EVERY_N_FRAMES == 0):
                background = update_bright_background(
                    background,
                    detector_gray,
                    background_decay_image,
                    track,
                )

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
                if calibrating:
                    status = "CALIBRATING %d/%d" % (
                        background_warmup, BACKGROUND_WARMUP_FRAMES
                    )
                else:
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
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        if media_initialized:
            MediaManager.deinit()


if __name__ == "__main__":
    main()
