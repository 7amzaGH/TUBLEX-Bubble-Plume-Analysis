"""
Offline TUBLEX demo.

Example:
python -m tublex.main_offline --video examples/sample.mp4 --output results/features.csv

With model:
python -m tublex.main_offline --video examples/sample.mp4 --model models/rf.joblib --metadata models/rf_metadata.json
"""

import argparse

from .config import get_config
from .video import process_video
from .classifier import load_and_predict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/tublex_features.csv")

    parser.add_argument("--profile", default="standard", choices=["standard", "sparse"])
    parser.add_argument("--use-roi", action="store_true")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=None)

    parser.add_argument("--label-start", type=float, default=None)
    parser.add_argument("--label-end", type=float, default=None)

    parser.add_argument("--model", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--threshold", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    roi = {
        "enabled": args.use_roi,
        "x_min": 0.38,
        "x_max": 0.62,
        "y_min": 0.00,
        "y_max": 0.65,
    }

    cfg = get_config(profile=args.profile, roi=roi)

    df = process_video(
        video_path=args.video,
        cfg=cfg,
        start_sec=args.start,
        end_sec=args.end,
        label_start=args.label_start,
        label_end=args.label_end,
        save_path=args.output,
    )

    print(f"Saved features: {args.output}")
    print(df.head())

    if args.model and args.metadata:
        pred_df = load_and_predict(
            df=df,
            model_path=args.model,
            metadata_path=args.metadata,
            threshold=args.threshold,
        )

        pred_path = args.output.replace(".csv", "_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        print(f"Saved predictions: {pred_path}")
        print(pred_df[["leak_probability", "predicted_class"]].head())


if __name__ == "__main__":
    main()