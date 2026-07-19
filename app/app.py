"""
Streamlit app for the credit risk project.

Matches src/train.py and src/predict.py: two explicitly named models for two
distinct personas, a business-driven decision threshold (not a silent 0.5),
and caps reused from training rather than recomputed on whatever the user
uploads.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

from src.feature_engineering import engineer_features, EMP_LENGTH_MAP

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Credit Risk Prediction", layout="wide", page_icon="💳")
st.title("💳 Credit Risk Prediction System")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

PERSONA_INFO = {
    "underwriting": {
        "label": "Underwriting — Screen a new applicant",
        "description": (
            "Use this when you do **not** yet know the loan's grade or interest "
            "rate — e.g. screening a brand-new applicant who hasn't been priced "
            "yet. Uses borrower-only information."
        ),
    },
    "investor": {
        "label": "Investor — Score an already-listed loan",
        "description": (
            "Use this when you **do** already know the loan's `sub_grade` and "
            "`int_rate` — e.g. an investor deciding which already-listed loan "
            "to fund. Includes LendingClub's own risk grade/pricing as input, "
            "so it cannot be used to screen a brand-new, ungraded applicant."
        ),
    },
}


# ===============================
# Load artifacts
# ===============================
@st.cache_resource
def load_persona_artifacts(persona: str):
    model_path = MODEL_DIR / f"credit_risk_model_{persona}.joblib"
    features_path = MODEL_DIR / f"expected_features_{persona}.joblib"
    caps_path = MODEL_DIR / "feature_caps.joblib"

    for p in (model_path, features_path, caps_path):
        if not p.exists():
            st.error(f"❌ Required file not found: {p}\n\nRun `python -m src.train` first.")
            st.stop()

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    caps = joblib.load(caps_path)
    return model, features, caps


@st.cache_data
def load_metrics():
    metrics_path = MODEL_DIR / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


metrics = load_metrics()

# ===============================
# Sidebar: persona + threshold
# ===============================
st.sidebar.header("Model & Decision Settings")

persona = st.sidebar.radio(
    "Which model do you need?",
    options=list(PERSONA_INFO.keys()),
    format_func=lambda p: PERSONA_INFO[p]["label"],
    index=0,
)
st.sidebar.caption(PERSONA_INFO[persona]["description"])

model, expected_features, caps = load_persona_artifacts(persona)

default_threshold = (
    metrics.get("investor_model", {}).get("optimal_threshold", 0.5)
    if persona == "investor" else 0.5
)
threshold = st.sidebar.slider(
    "Decision threshold (flag as High Risk above this probability)",
    min_value=0.05, max_value=0.95, value=float(default_threshold), step=0.05,
    help=(
        "Defaults to the cost-minimizing threshold found during training "
        "(see model_metrics.json), not a fixed 0.5 cutoff. Adjust if your "
        "risk tolerance differs from the illustrative cost assumptions used "
        "in training — see the Model Performance tab."
    ),
)

st.sidebar.divider()
persona_metrics = metrics.get(f"{persona}_model", {})
if persona_metrics.get("test_roc_auc"):
    st.sidebar.metric("Model Test ROC-AUC", f"{persona_metrics['test_roc_auc']:.4f}")

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Batch Prediction (CSV)", "🧍 Single Applicant", "📈 Model Performance", "ℹ️ Information"]
)

# ------------- Tab 1: Batch prediction -------------
with tab1:
    st.markdown("### 📥 Upload Loan Data")
    st.info(
        f"Using the **{PERSONA_INFO[persona]['label']}** model. "
        "Upload a CSV matching the schema of the training dataset."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, low_memory=False)
            input_df = input_df.drop(columns=["loan_status", "target"], errors="ignore")

            # Reuse training-time caps -- do not recompute from this upload,
            # which could be small and produce unstable caps (see
            # src/feature_engineering.py).
            input_df = engineer_features(input_df, caps=caps)

            missing_cols = set(expected_features) - set(input_df.columns)
            extra_cols = set(input_df.columns) - set(expected_features)

            if missing_cols:
                st.error(
                    f"❌ Missing required columns for the **{persona}** model:\n{sorted(missing_cols)}\n\n"
                    + ("Try the Underwriting model instead if you don't have sub_grade/int_rate."
                       if persona == "investor" else "")
                )
                st.stop()

            if extra_cols:
                st.warning(f"⚠️ Extra columns detected and ignored:\n{sorted(extra_cols)}")

            X = input_df[expected_features]

            with st.spinner("Scoring records..."):
                probs = model.predict_proba(X)[:, 1]
                preds = (probs >= threshold).astype(int)

            results = input_df.copy()
            results["Default Probability"] = probs
            results["Prediction Label"] = np.where(
                preds == 1, "High Risk (Default)", "Low Risk (Repay)"
            )

            st.success("✅ Prediction Completed Successfully!")
            st.markdown("---")
            st.subheader("📈 Prediction Summary")

            total = len(results)
            defaulters = int(preds.sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", f"{total:,}")
            m2.metric("Predicted High Risk", f"{defaulters:,}", f"{defaulters/total*100:.1f}% of total")
            m3.metric("Predicted Low Risk", f"{total - defaulters:,}")
            m4.metric("Avg Default Probability", f"{probs.mean():.2%}")

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(
                    names=["Low Risk", "High Risk"],
                    values=[total - defaulters, defaulters],
                    title="Loan Risk Distribution", hole=0.4,
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                fig_hist = px.histogram(
                    results, x="Default Probability", nbins=30,
                    color="Prediction Label", title="Probability Distribution",
                    color_discrete_map={"Low Risk (Repay)": "#2ecc71", "High Risk (Default)": "#e74c3c"},
                )
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="gray",
                                    annotation_text=f"Threshold = {threshold}")
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.subheader("Detailed Results")

            # Styling is only applied to a preview -- pandas Styler does not
            # scale to large uploads and would slow or hang on a realistic
            # batch size. Full unstyled results are always in the download.
            PREVIEW_ROWS = 500
            if total > PREVIEW_ROWS:
                st.caption(f"Showing a styled preview of the first {PREVIEW_ROWS:,} of {total:,} rows. "
                           "Full results are in the download below.")

            def highlight_risk(row):
                if row["Prediction Label"] == "High Risk (Default)":
                    return ["background-color: rgba(231, 76, 60, 0.2)"] * len(row)
                return [""] * len(row)

            preview = results.head(PREVIEW_ROWS)
            st.dataframe(preview.style.apply(highlight_risk, axis=1), use_container_width=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Predictions (CSV)", data=csv,
                file_name=f"credit_risk_predictions_{persona}.csv", mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Something went wrong while scoring this file: {e}")
            st.caption("Check that the uploaded CSV matches the expected schema (see the Information tab).")
    else:
        st.write("👆 Upload a CSV dataset to get started.")

# ------------- Tab 2: Single applicant -------------
with tab2:
    st.markdown(f"### 🧍 Score a Single Applicant — {PERSONA_INFO[persona]['label']}")

    with st.form("single_applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 15000, step=500)
            term = st.selectbox("Term", [" 36 months", " 60 months"])
            installment = st.number_input("Monthly installment ($)", 10.0, 2000.0, 450.0)
            annual_inc = st.number_input("Annual income ($)", 1000.0, 1_000_000.0, 60000.0, step=1000.0)
            emp_length = st.selectbox("Employment length", list(EMP_LENGTH_MAP.keys()))
        with c2:
            home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            verification_status = st.selectbox(
                "Verification status", ["Verified", "Source Verified", "Not Verified"]
            )
            purpose = st.selectbox(
                "Purpose", ["debt_consolidation", "credit_card", "home_improvement",
                            "small_business", "major_purchase", "medical", "other"]
            )
            dti = st.number_input("DTI (%)", 0.0, 60.0, 18.0)
            application_type = st.selectbox("Application type", ["Individual", "Joint App"])
        with c3:
            fico_low = st.number_input("FICO range low", 300, 850, 690)
            fico_high = st.number_input("FICO range high", 300, 850, 694)
            delinq_2yrs = st.number_input("Delinquencies (2yr)", 0, 20, 0)
            open_acc = st.number_input("Open accounts", 0, 50, 10)
            pub_rec = st.number_input("Public records", 0, 10, 0)
            revol_bal = st.number_input("Revolving balance ($)", 0.0, 200000.0, 8000.0)
            revol_util = st.number_input("Revolving utilization (%)", 0.0, 150.0, 40.0)
            total_acc = st.number_input("Total accounts", 1, 100, 20)

        if persona == "investor":
            sub_grade = st.selectbox(
                "Sub-grade", [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
            )
            int_rate = st.number_input("Interest rate (%)", 5.0, 35.0, 13.0)

        submitted = st.form_submit_button("Score Applicant")

    if submitted:
        try:
            row = {
                "loan_amnt": loan_amnt, "term": term, "installment": installment,
                "annual_inc": annual_inc, "emp_length": emp_length,
                "home_ownership": home_ownership, "verification_status": verification_status,
                "purpose": purpose, "dti": dti, "application_type": application_type,
                "fico_range_low": fico_low, "fico_range_high": fico_high,
                "delinq_2yrs": delinq_2yrs, "open_acc": open_acc, "pub_rec": pub_rec,
                "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
                # int_rate is required by engineer_features regardless of persona
                # (it's cleaned upstream of the feature split); only actually
                # used as a MODEL feature when persona == "investor".
                "int_rate": int_rate if persona == "investor" else 15.0,
            }
            if persona == "investor":
                row["sub_grade"] = sub_grade

            single_df = pd.DataFrame([row])
            single_df = engineer_features(single_df, caps=caps)
            X_single = single_df[expected_features]
            prob = model.predict_proba(X_single)[0, 1]
            label = "High Risk (Default)" if prob >= threshold else "Low Risk (Repay)"

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Default Probability", f"{prob:.2%}")
            col2.metric("Prediction", label)
            st.progress(min(float(prob), 1.0))

            if persona == "underwriting":
                st.caption(
                    "Note: interest rate was not used as an input for this prediction "
                    "(Underwriting model is borrower-only)."
                )

        except Exception as e:
            st.error(f"❌ Could not score this applicant: {e}")

# ------------- Tab 3: Model performance -------------
with tab3:
    st.subheader("Model Performance")
    if not metrics:
        st.warning("No model_metrics.json found — run `python -m src.train` first.")
    else:
        st.write(f"**Selected model type:** {metrics.get('chosen_model_type', 'n/a')}")
        st.write(f"**Chronological split date:** {metrics.get('split_date', 'n/a')}")

        st.markdown("#### Baselines vs. trained models")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Majority-class baseline (accuracy)",
                   metrics.get("baselines", {}).get("majority_class_accuracy", "n/a"))
        b2.metric("sub_grade-alone baseline (AUC)",
                   metrics.get("baselines", {}).get("subgrade_alone_auc", "n/a"))
        inv = metrics.get("investor_model", {})
        uw = metrics.get("underwriting_model", {})
        b3.metric("Investor Model (AUC)",
                   f"{inv.get('test_roc_auc', 0):.4f}" if inv.get("test_roc_auc") else "n/a")
        b4.metric("Underwriting Model (AUC)",
                   f"{uw.get('test_roc_auc', 0):.4f}" if uw.get("test_roc_auc") else "n/a")

        st.caption(
            "The gap between the Investor and Underwriting AUC reflects how much of the "
            "Investor Model's predictive power comes from LendingClub's own sub_grade/int_rate "
            "pricing, rather than new borrower-level signal."
        )

        if inv.get("brier_score") is not None:
            st.write(f"**Investor Model Brier score (calibration):** {inv['brier_score']:.4f} "
                     "— lower is better; measures whether predicted probabilities match observed rates.")

        if inv.get("top_permutation_importances"):
            st.markdown("#### Top feature importances (permutation, Investor Model)")
            imp_df = pd.DataFrame(
                list(inv["top_permutation_importances"].items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=False)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("⚠️ Threshold cost assumptions (illustrative, not verified figures)"):
            st.json(inv.get("threshold_cost_assumptions", {}))

# ------------- Tab 4: Info -------------
with tab4:
    st.subheader("Model Information")
    st.write(
        "This system uses **two separate models**, each valid for a different persona — "
        "see the sidebar for which one to use."
    )
    for p, info in PERSONA_INFO.items():
        st.markdown(f"**{info['label']}**")
        st.caption(info["description"])

    st.write(f"**Required features for the currently selected ({persona}) model, "
              f"{len(expected_features)} columns:**")
    st.code(", ".join(expected_features), language="text")

    st.markdown("---")
    st.markdown(
        "**Known limitations:** trained only on LendingClub's approved-loan population "
        "(rejected applicants are not represented); expected-loss/threshold figures use "
        "illustrative cost assumptions, not verified institutional data. See "
        "`notebooks/5_business_insights.ipynb` for the full limitations section."
    )