"""
Adaptive preprocessing for underwater bubble frames.
"""

import cv2
import numpy as np


def crop_roi(frame, roi_cfg):
    """Crop frame using normalized ROI coordinates."""
    h, w = frame.shape[:2]

    if not roi_cfg.get("enabled", True):
        return frame.copy(), (0, 0, w, h)

    x1 = int(w * roi_cfg["x_min"])
    x2 = int(w * roi_cfg["x_max"])
    y1 = int(h * roi_cfg["y_min"])
    y2 = int(h * roi_cfg["y_max"])

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def select_water_params(roi_frame, cfg):
    """Choose bright-water or dark-water parameters using HSV V mean."""
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    v_mean = float(np.mean(v_channel))

    water_type = "bright_water" if v_mean > cfg["water_threshold"] else "dark_water"
    params = cfg[water_type]

    return params, water_type, v_mean, v_channel


def preprocess_frame(frame, cfg):
    """
    Apply ROI cropping, clutter masking, CLAHE, and Gaussian smoothing.
    """
    roi_frame, roi_box = crop_roi(frame, cfg["roi"])
    params, water_type, v_mean, v_channel = select_water_params(roi_frame, cfg)

    _, clutter_mask = cv2.threshold(
        v_channel,
        params["clutter_v_threshold"],
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel = np.ones((params["dilation"], params["dilation"]), np.uint8)
    search_mask = cv2.bitwise_not(cv2.dilate(clutter_mask, kernel, iterations=1))

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=params["clahe_clip"],
        tileGridSize=params["clahe_tile"],
    )

    enhanced = clahe.apply(gray)
    masked = cv2.bitwise_and(enhanced, search_mask)
    blurred = cv2.GaussianBlur(masked, params["blur"], 2)

    return {
        "image": blurred,
        "roi_frame": roi_frame,
        "roi_box": roi_box,
        "water_type": water_type,
        "water_v_mean": v_mean,
        "params": params,
    }