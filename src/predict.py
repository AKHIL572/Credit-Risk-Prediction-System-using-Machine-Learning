"""
Prediction script for the credit risk project.

Usage:
    python -m src.predict <input_csv_path> --model investor
    python -m src.predict <input_csv_path> --model underwriting

--model investor:
    Requires sub_grade and int_rate to already be known -- valid for scoring
    an ALREADY-LISTED loan (e.g. an investor deciding which loan to fund).
--model underwriting:
    Uses borrower-only features -- valid for screening a NEW applicant who
    has not yet been graded/priced by LendingClub. This is the only one of
    the two that can legitimately be used for that purpose.

If you're not sure which one you need: if you don't already know the loan's
grade and interest rate, use --model underwriting.
"""

import os
import sys
import argparse
import json
import joblib
import pandas as pd

from src.feature_engineering import engineer_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def load_artifacts(persona: str):
    model_path = os.path.join(MODEL_DIR, f"credit_risk_model_{persona}.joblib")
    features_path = os.path.join(
        MODEL_DIR, f"expected_features_{persona}.joblib")
    caps_path = os.path.join(MODEL_DIR, "feature_caps.joblib")
    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")

    for p in (model_path, features_path, caps_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required artifact not found: {p}. Run src/train.py first.")

    model = joblib.load(model_path)
    expected_features = joblib.load(features_path)
    caps = joblib.load(caps_path)

    optimal_threshold = 0.5
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        optimal_threshold = metrics.get(
            "investor_model", {}).get("optimal_threshold", 0.5)

    return model, expected_features, caps, optimal_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Score loans with a trained credit risk model.")
    parser.add_argument("input_csv", help="Path to input CSV file.")
    parser.add_argument(
        "--model", choices=["investor", "underwriting"], default="underwriting",
        help="Which persona's model to use. Default: underwriting (safer default -- "
             "does not require sub_grade/int_rate to already be known)."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the decision threshold. Defaults to the optimal threshold "
             "found during training (see model_metrics.json), or 0.5 if unavailable."
    )
    args = parser.parse_args()

    model, expected_features, caps, optimal_threshold = load_artifacts(
        args.model)
    threshold = args.threshold if args.threshold is not None else optimal_threshold

    df = pd.read_csv(args.input_csv, low_memory=False)
    df = df.drop(columns=["loan_status", "target"], errors="ignore")

    # Reuse the caps computed on TRAINING data -- do not recompute from this
    # (possibly small) input batch, which would produce unstable, inconsistent
    # caps between runs (see src/feature_engineering.py module docstring).
    df = engineer_features(df, caps=caps)

    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for the '{args.model}' model: {missing_cols}\n"
            f"If you don't have sub_grade/int_rate for these records, use --model underwriting instead."
        )

    df_scored = df[expected_features].copy()
    df["default_probability"] = model.predict_proba(df_scored)[:, 1]
    df["default_prediction"] = (
        df["default_probability"] >= threshold).astype(int)
    df["model_used"] = args.model
    df["decision_threshold_used"] = threshold

    output_path = os.path.join(
        PROJECT_ROOT, "data", "processed", f"output_predictions_{args.model}.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Model used: {args.model} | Threshold: {threshold}")


if __name__ == "__main__":
    main()
