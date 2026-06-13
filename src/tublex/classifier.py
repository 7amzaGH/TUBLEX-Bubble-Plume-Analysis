"""
Model loading and prediction utilities for TUBLEX.

Works with sklearn-compatible models such as RandomForestClassifier
and XGBClassifier saved with joblib.
"""

import json
from pathlib import Path

import joblib
import numpy as np


def load_model(model_path):
    """Load a joblib model."""
    return joblib.load(model_path)


def load_metadata(metadata_path):
    """Load model metadata JSON."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_columns(metadata_path):
    """Load expected feature column order."""
    metadata = load_metadata(metadata_path)
    return metadata["feature_columns"]


def prepare_features(df, feature_columns):
    """Select feature columns in the correct order."""
    missing = [c for c in feature_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return df[feature_columns]


def get_positive_class_index(model, positive_class=1):
    """Find probability column index for the positive class."""
    classes = getattr(model, "classes_", None)

    if classes is None:
        return 1

    classes = list(classes)

    if positive_class not in classes:
        raise ValueError(f"Positive class {positive_class} not found in {classes}")

    return classes.index(positive_class)


def predict_features(df, model, feature_columns, threshold=0.5, positive_class=1):
    """Predict leak probability and binary label."""
    X = prepare_features(df, feature_columns)

    positive_idx = get_positive_class_index(model, positive_class)
    leak_prob = model.predict_proba(X)[:, positive_idx]
    pred_label = (leak_prob >= threshold).astype(int)

    out = df.copy()
    out["leak_probability"] = leak_prob
    out["predicted_label"] = pred_label
    out["predicted_class"] = np.where(pred_label == 1, "leak", "non_leak")

    return out


def load_and_predict(df, model_path, metadata_path, threshold=None):
    """Load model + metadata and run prediction."""
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)

    feature_columns = metadata["feature_columns"]
    threshold = metadata.get("decision_threshold", 0.5) if threshold is None else threshold
    positive_class = metadata.get("positive_class", 1)

    return predict_features(
        df=df,
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        positive_class=positive_class,
    )


def save_metadata(
    output_path,
    feature_columns,
    model_name="TUBLEX Model",
    model_type="RandomForestClassifier",
    threshold=0.5,
    positive_class=1,
):
    """Save model metadata JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "positive_class": positive_class,
        "decision_threshold": threshold,
        "feature_columns": list(feature_columns),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)