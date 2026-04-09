import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide",
    page_icon="💳"
)

st.title("💳 Credit Risk Prediction System")
st.markdown("Predict the probability of loan default using a trained Machine Learning model.")

# ===============================
# Paths (ROBUST)
# ===============================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / ".." / "models" / "credit_risk_model.joblib"
FEATURES_PATH = BASE_DIR / ".." / "models" / "expected_features.joblib"

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
tab1, tab2 = st.tabs(["📊 Batch Prediction (CSV)", "ℹ️ Information"])

with tab1:
    st.markdown("### 📥 Upload Customer Data")
    st.info("Please upload a CSV file matching the schema of the training dataset. The model will analyze the records and provide a default prediction for each loan.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
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
        with st.spinner("Analyzing data and generating predictions..."):
            preds = model.predict(input_df)
            probs = model.predict_proba(input_df)[:, 1]

        results = input_df.copy()
        
        # Map predictions to human readable format
        results["Prediction Label"] = ["High Risk (Default)" if p == 1 else "Low Risk (Repay)" for p in preds]
        results["Default Probability"] = probs
        
        st.success("✅ Prediction Completed Successfully!")
        st.markdown("---")

        # ===============================
        # Dashboard / Display results
        # ===============================
        st.subheader("📈 Prediction Summary")
        
        total_records = len(results)
        defaulters = int((preds == 1).sum())
        non_defaulters = total_records - defaulters
        avg_prob = probs.mean()
        
        # Metrics 
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{total_records:,}")
        m2.metric("Predicted Defaulters", f"{defaulters:,}", f"{(defaulters/total_records)*100:.1f}% of total", delta_color="inverse")
        m3.metric("Predicted Safe Loans", f"{non_defaulters:,}")
        m4.metric("Avg Default Probability", f"{avg_prob:.2%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                names=["Low Risk (Repay)", "High Risk (Default)"], 
                values=[non_defaulters, defaulters],
                title="Loan Risk Distribution",
                hole=0.4,
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            fig_hist = px.histogram(
                results, 
                x="Default Probability", 
                title="Probability Density",
                nbins=30,
                color="Prediction Label",
                color_discrete_map={"Low Risk (Repay)": "#2ecc71", "High Risk (Default)": "#e74c3c"}
            )
            fig_hist.update_layout(margin=dict(t=40, b=0, l=0, r=0), xaxis_title="Probability of Default", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")

        st.subheader("Detailed Results")
        
        # Helper function for conditional styling
        def highlight_risk(row):
            if row["Prediction Label"] == "High Risk (Default)":
                return ['background-color: rgba(231, 76, 60, 0.2)'] * len(row)
            return [''] * len(row)
            
        st.dataframe(results.style.apply(highlight_risk, axis=1), use_container_width=True)

        # ===============================
        # Download
        # ===============================
        st.markdown("<br>", unsafe_allow_html=True)
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Predictions (CSV)",
            data=csv,
            file_name="credit_risk_predictions.csv",
            mime="text/csv"
        )
    else:
        st.write("👆 Upload a CSV dataset to get started.")

with tab2:
    st.subheader("Model Information")
    st.write(
        "This Credit Risk Prediction model analyzes various financial and personal indicators to predict the probability of a loan default."
    )
    st.write(f"**Required Features ({len(expected_features)} columns):**")
    st.code(", ".join(expected_features), language="text")
