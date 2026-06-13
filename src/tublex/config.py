"""
Central configuration for the TUBLEX pipeline.
"""

from copy import deepcopy


FEATURE_COLUMNS = [
    "mean_bubble_count",
    "max_bubble_count",
    "std_bubble_count",
    "continuity_ratio",
    "mean_vertical_chain",
    "temporal_variance",
    "prev3_mean_bubble_count",
    "prev3_std_bubble_count",
    "prev3_continuity_ratio",
    "prev3_mean_vertical_chain",
    "leak_evolution",
]


DEFAULT_ROI = {
    "enabled": False,
    "x_min": 0.00,
    "x_max": 1.00,
    "y_min": 0.00,
    "y_max": 1.00,
}

CONTROLLED_PLUME_ROI = {
    "enabled": True,
    "x_min": 0.38,
    "x_max": 0.62,
    "y_min": 0.00,
    "y_max": 0.65,
}


PROFILES = {
    "standard": {
        "water_threshold": 100,
        "output_fps": 10,
        "window_sec": 1.0,
        "history_windows": 3,
        "decision_threshold": 0.50,

        "bright_water": {
            "clahe_clip": 2.0,
            "clahe_tile": (8, 8),
            "blur": (9, 9),
            "clutter_v_threshold": 50,
            "dilation": 20,
            "min_area": 20,
            "max_area": 5000,
            "min_circularity": 0.60,
        },

        "dark_water": {
            "clahe_clip": 2.0,
            "clahe_tile": (8, 8),
            "blur": (7, 7),
            "clutter_v_threshold": 16,
            "dilation": 15,
            "min_area": 5,
            "max_area": 500,
            "min_circularity": 0.70,
        },

        "dbscan": {
            "eps": 50,
            "min_samples": 2,
        },

        "vertical_chain": {
            "x_tolerance": 30,
            "y_gap_ratio": 0.20,
            "empty_value": 0,
        },
    },

    "sparse": {
        "water_threshold": 100,
        "output_fps": 10,
        "window_sec": 1.0,
        "history_windows": 3,
        "decision_threshold": 0.50,

        "bright_water": {
            "clahe_clip": 2.0,
            "clahe_tile": (8, 8),
            "blur": (9, 9),
            "clutter_v_threshold": 50,
            "dilation": 20,
            "min_area": 10,
            "max_area": 5000,
            "min_circularity": 0.30,
        },

        "dark_water": {
            "clahe_clip": 2.0,
            "clahe_tile": (8, 8),
            "blur": (7, 7),
            "clutter_v_threshold": 16,
            "dilation": 15,
            "min_area": 5,
            "max_area": 500,
            "min_circularity": 0.30,
        },

        "dbscan": {
            "eps": 50,
            "min_samples": 1,
        },

        "vertical_chain": {
            "x_tolerance": 45,
            "y_gap_ratio": 0.25,
            "empty_value": 0,
        },
    },
}


def get_config(profile="standard", roi=None, external_cue=None):
    """
    Return a pipeline configuration.

    profile:
        "standard" is the default publication-ready setting.
        "sparse" is more permissive for few-bubble scenarios.

    external_cue:
        Optional cue from another sensor. It only changes the visual profile;
        it is not used as a classifier feature.
    """
    if external_cue in ["suspicious", "alert"]:
        profile = "sparse"

    if profile not in PROFILES:
        raise ValueError("profile must be 'standard' or 'sparse'.")

    cfg = deepcopy(PROFILES[profile])
    cfg["profile"] = profile
    cfg["roi"] = deepcopy(DEFAULT_ROI if roi is None else roi)

    return cfg