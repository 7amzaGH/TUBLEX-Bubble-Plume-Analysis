"""
Video reading, sampling, and window-level feature generation.
"""

import re
from pathlib import Path

import cv2
import pandas as pd

from .config import get_config
from .features_extraction import (
    extract_frame_features,
    aggregate_window,
    add_temporal_features,
    reorder_columns,
)


def parse_video_metadata(video_path):
    """Extract gauge and pressure from filename when available."""
    name = Path(video_path).stem

    g_match = re.search(r"G(\d+)", name)
    psi_match = re.search(r"(\d+)\s*PSI", name.upper())

    g = int(g_match.group(1)) if g_match else None
    psi = int(psi_match.group(1)) if psi_match else None

    return g, psi


def get_label(time_sec, label_start=None, label_end=None):
    """Return label using leak interval if provided."""
    if label_start is None or label_end is None:
        return None

    return int(label_start <= time_sec <= label_end)


def process_video(
    video_path,
    cfg=None,
    start_sec=0,
    end_sec=None,
    label_start=None,
    label_end=None,
    source_video_id=None,
    save_path=None,
    debug=False,
):
    """
    Process a video into 1-second TUBLEX window features.

    Each output row represents one fixed-duration temporal window.
    The column time_sec represents the decision time at the end of that window.
    """
    cfg = get_config() if cfg is None else cfg

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = cfg["output_fps"]
    frame_step = max(1, round(original_fps / output_fps))
    window_size = int(output_fps * cfg["window_sec"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * original_fps)
    end_frame = int(end_sec * original_fps) if end_sec is not None else total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    g, psi = parse_video_metadata(video_path)
    source_video_id = source_video_id or Path(video_path).stem

    rows = []
    buffer = []
    window_id = 0
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_features = extract_frame_features(frame, cfg, debug=debug)
            buffer.append(frame_features)

            if len(buffer) == window_size:
                time_sec = start_sec + (window_id + 1) * cfg["window_sec"]

                row = aggregate_window(buffer)
                row["source_video_id"] = source_video_id
                row["window_id"] = window_id
                row["time_sec"] = time_sec
                row["sample_id"] = f"{source_video_id}_w{window_id:04d}"

                if g is not None:
                    row["g"] = g

                if psi is not None:
                    row["psi"] = psi

                label = get_label(time_sec, label_start, label_end)

                if label is not None:
                    row["label"] = label

                rows.append(row)
                buffer = []
                window_id += 1

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = add_temporal_features(df, history=cfg["history_windows"])
        df = reorder_columns(df)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df