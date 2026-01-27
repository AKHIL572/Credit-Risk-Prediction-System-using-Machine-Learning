import os
import sys
import joblib
import pandas as pd


# =========================
# 1. Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "expected_features.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_predictions.csv")


# =========================
# 2. Load model & metadata
# =========================
model = joblib.load(MODEL_PATH)
expected_features = joblib.load(FEATURES_PATH)


# =========================
# 3. Read input
# =========================
if len(sys.argv) < 2:
    raise ValueError("Usage: python predict.py <input_csv_path>")

input_path = sys.argv[1]
df = pd.read_csv(input_path, low_memory=False)

# Remove target columns if present
df = df.drop(columns=["loan_status", "target"], errors="ignore")


# =========================
# 4. Validate schema
# =========================
missing_cols = set(expected_features) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Reorder columns
df = df[expected_features]


# =========================
# 5. Predict
# =========================
df["default_prediction"] = model.predict(df)
df["default_probability"] = model.predict_proba(df)[:, 1]


# =========================
# 6. Save output
# =========================
df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to: {OUTPUT_PATH}")
