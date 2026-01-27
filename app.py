import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide"
)

st.title("💳 Credit Risk Prediction System")
st.write("Predict the probability of loan default using a trained ML model.")

# ===============================
# Paths (ROBUST)
# ===============================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Models" / "credit_risk_model.joblib"
FEATURES_PATH = BASE_DIR / "Models" / "expected_features.joblib"

# ===============================
# Load model & metadata
# ===============================


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()

    if not FEATURES_PATH.exists():
        st.error(f"❌ Feature list not found at: {FEATURES_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    expected_features = joblib.load(FEATURES_PATH)

    return model, expected_features


model, expected_features = load_artifacts()

# ===============================
# Upload input data
# ===============================
st.subheader("📥 Upload Customer Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV file (same schema as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, low_memory=False)

    # Remove target columns if user included them
    input_df = input_df.drop(
        columns=["loan_status", "target"],
        errors="ignore"
    )

    # ===============================
    # Validate schema
    # ===============================
    missing_cols = set(expected_features) - set(input_df.columns)
    extra_cols = set(input_df.columns) - set(expected_features)

    if missing_cols:
        st.error(f"❌ Missing required columns:\n{missing_cols}")
        st.stop()

    if extra_cols:
        st.warning(
            f"⚠️ Extra columns detected and ignored:\n{extra_cols}"
        )
        input_df = input_df.drop(columns=list(extra_cols))

    # Reorder columns exactly as training
    input_df = input_df[expected_features]

    # ===============================
    # Prediction
    # ===============================
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]

    results = input_df.copy()
    results["Default Prediction"] = preds
    results["Default Probability"] = probs

    # ===============================
    # Display results
    # ===============================
    st.subheader("📊 Prediction Results")
    st.dataframe(results, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(results))
    col2.metric("Predicted Defaulters", int((preds == 1).sum()))

    # ===============================
    # Download
    # ===============================
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions",
        data=csv,
        file_name="credit_risk_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV file to get predictions")
