"""
Simple live CPU-time TUBLEX demo.

Example:
python -m tublex.main_live --source 0 --model models/rf.joblib --metadata models/rf_metadata.json
"""

import argparse
import time

import cv2
import pandas as pd

from .config import get_config
from .features_extraction import (
    extract_frame_features,
    aggregate_window,
    add_temporal_features,
)
from .classifier import load_model, load_metadata, predict_features


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="0")
    parser.add_argument("--profile", default="standard", choices=["standard", "sparse"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--threshold", type=float, default=None)

    return parser.parse_args()


def open_source(source):
    """Open webcam index or video path."""
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))

    return cv2.VideoCapture(source)


def main():
    args = parse_args()

    cfg = get_config(profile=args.profile)

    model = load_model(args.model)
    metadata = load_metadata(args.metadata)

    feature_columns = metadata["feature_columns"]
    threshold = args.threshold or metadata.get("decision_threshold", 0.5)

    cap = open_source(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = cfg["output_fps"]

    frame_step = max(1, round(fps / cfg["output_fps"]))
    window_size = int(cfg["output_fps"] * cfg["window_sec"])

    frame_buffer = []
    window_rows = []
    frame_idx = 0

    last_label = "waiting"
    last_prob = 0.0
    last_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_features = extract_frame_features(frame, cfg)
            frame_buffer.append(frame_features)

            if len(frame_buffer) == window_size:
                row = aggregate_window(frame_buffer)
                window_rows.append(row)

                df = pd.DataFrame(window_rows)
                df = add_temporal_features(df, history=cfg["history_windows"])

                latest = df.tail(1)
                pred = predict_features(
                    latest,
                    model=model,
                    feature_columns=feature_columns,
                    threshold=threshold,
                    positive_class=metadata.get("positive_class", 1),
                )

                last_prob = float(pred["leak_probability"].iloc[0])
                last_label = str(pred["predicted_class"].iloc[0])

                frame_buffer = []

        now = time.time()
        fps_live = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        text = f"TUBLEX: {last_label} | P(leak)={last_prob:.3f} | FPS={fps_live:.1f}"
        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("TUBLEX Live Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()