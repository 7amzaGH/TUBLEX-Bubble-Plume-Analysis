"""
Frame-level and temporal-window feature extraction.
"""

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .preprocessing import preprocess_frame
from .detection import detect_frame_bubbles


def extract_frame_features(frame, cfg, debug=False):
    """Extract bubble features from one BGR frame."""
    prep = preprocess_frame(frame, cfg)
    det = detect_frame_bubbles(prep, cfg)

    row = {
        "bubble_count": det["bubble_count"],
        "vertical_chain": det["vertical_chain"],
        "frame_has_bubbles": det["frame_has_bubbles"],
    }

    if debug:
        row.update({
            "raw_bubble_count": det["raw_count"],
            "n_clusters": det["n_clusters"],
            "water_type": prep["water_type"],
            "water_v_mean": prep["water_v_mean"],
        })

    return row


def aggregate_window(frame_rows):
    """Aggregate sampled frame rows into one temporal window."""
    df = pd.DataFrame(frame_rows)

    counts = df["bubble_count"].astype(float).values
    chains = df["vertical_chain"].astype(float).values
    presence = df["frame_has_bubbles"].astype(float).values

    return {
        "mean_bubble_count": float(np.mean(counts)),
        "max_bubble_count": float(np.max(counts)),
        "std_bubble_count": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
        "continuity_ratio": float(np.mean(presence)),
        "mean_vertical_chain": float(np.mean(chains)),
        "temporal_variance": float(np.var(counts, ddof=1)) if len(counts) > 1 else 0.0,
    }


def add_temporal_features(df, history=3):
    """
    Add causal 3-window memory features.
    Only previous windows are used.
    """
    df = df.copy()

    df["prev3_mean_bubble_count"] = (
        df["mean_bubble_count"].shift(1).rolling(history, min_periods=1).mean().fillna(0)
    )

    df["prev3_std_bubble_count"] = (
        df["std_bubble_count"].shift(1).rolling(history, min_periods=1).mean().fillna(0)
    )

    df["prev3_continuity_ratio"] = (
        df["continuity_ratio"].shift(1).rolling(history, min_periods=1).mean().fillna(0)
    )

    df["prev3_mean_vertical_chain"] = (
        df["mean_vertical_chain"].shift(1).rolling(history, min_periods=1).mean().fillna(0)
    )

    df["leak_evolution"] = (
        df["mean_bubble_count"] - df["prev3_mean_bubble_count"]
    )

    return df


def get_model_features(df):
    """Return model features in the correct order."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return df[FEATURE_COLUMNS]


def reorder_columns(df):
    """Place metadata first, then model features, then other columns."""
    metadata = [
    	"sample_id",
    	"source_video_id",
    	"window_id",
    	"time_sec",
    	"g",
    	"psi",
    	"label",
    ]

    first = [c for c in metadata if c in df.columns]
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    others = [c for c in df.columns if c not in first + features]

    return df[first + features + others]