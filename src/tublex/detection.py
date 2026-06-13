"""
Bubble detection and spatial filtering.
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def create_blob_detector(params):
    """Create OpenCV blob detector from selected water parameters."""
    blob_params = cv2.SimpleBlobDetector_Params()

    blob_params.filterByColor = True
    blob_params.blobColor = 255

    blob_params.filterByArea = True
    blob_params.minArea = params["min_area"]
    blob_params.maxArea = params["max_area"]

    blob_params.filterByCircularity = True
    blob_params.minCircularity = params["min_circularity"]

    blob_params.filterByConvexity = False
    blob_params.filterByInertia = False

    return cv2.SimpleBlobDetector_create(blob_params)


def detect_bubbles(preprocessed_image, params):
    """Detect raw bubble candidates."""
    detector = create_blob_detector(params)
    return list(detector.detect(preprocessed_image))


def filter_bubbles_dbscan(keypoints, dbscan_cfg):
    """Remove isolated detections using DBSCAN."""
    if len(keypoints) == 0:
        return [], 0

    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    labels = DBSCAN(
        eps=dbscan_cfg["eps"],
        min_samples=dbscan_cfg["min_samples"],
    ).fit_predict(points)

    filtered = [kp for kp, label in zip(keypoints, labels) if label != -1]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return filtered, n_clusters


def compute_vertical_chain(keypoints, roi_height, chain_cfg):
    """
    Estimate the longest vertical bubble chain.
    Empty frames return 0.
    """
    if len(keypoints) == 0:
        return chain_cfg["empty_value"]

    sorted_kps = sorted(keypoints, key=lambda kp: kp.pt[1])

    max_chain = 1
    current_chain = 1
    max_y_gap = roi_height * chain_cfg["y_gap_ratio"]

    for prev_kp, curr_kp in zip(sorted_kps[:-1], sorted_kps[1:]):
        dx = abs(curr_kp.pt[0] - prev_kp.pt[0])
        dy = curr_kp.pt[1] - prev_kp.pt[1]

        if dx < chain_cfg["x_tolerance"] and 0 < dy < max_y_gap:
            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            current_chain = 1

    return int(max_chain)


def keypoints_to_points(keypoints, roi_box=None):
    """
    Convert keypoints to simple point dictionaries.

    If roi_box is given, coordinates are shifted back to full-frame space.
    """
    x_offset, y_offset = (0, 0) if roi_box is None else (roi_box[0], roi_box[1])

    return [
        {
            "x": float(kp.pt[0] + x_offset),
            "y": float(kp.pt[1] + y_offset),
            "size": float(kp.size),
        }
        for kp in keypoints
    ]


def detect_frame_bubbles(preprocess_result, cfg):
    """Run full bubble detection for one preprocessed frame."""
    raw_kps = detect_bubbles(
        preprocess_result["image"],
        preprocess_result["params"],
    )

    filtered_kps, n_clusters = filter_bubbles_dbscan(
        raw_kps,
        cfg["dbscan"],
    )

    vertical_chain = compute_vertical_chain(
        filtered_kps,
        preprocess_result["image"].shape[0],
        cfg["vertical_chain"],
    )

    return {
        "raw_keypoints": raw_kps,
        "keypoints": filtered_kps,
        "points": keypoints_to_points(filtered_kps, preprocess_result["roi_box"]),
        "raw_count": len(raw_kps),
        "bubble_count": len(filtered_kps),
        "n_clusters": n_clusters,
        "vertical_chain": vertical_chain,
        "frame_has_bubbles": int(len(filtered_kps) > 0),
    }