# 💳 Credit Risk Prediction System

An end-to-end machine learning project that predicts loan default probability
using historical LendingClub loan data — from data understanding and EDA
through preprocessing, modeling, business insights, and deployment via
Streamlit and Power BI.

---

## 📌 Project Overview

Loan investors and financial institutions need to assess the likelihood that
a borrower will default before capital is committed. This project trains
**two** models, each for a distinct, explicitly named use case:

- **Investor Risk Model** — scores an already-listed loan (uses LendingClub's
  own `sub_grade`/`int_rate`, valid only once those exist).
- **Underwriting Screening Model** — screens a brand-new applicant who has
  not yet been graded or priced (borrower-only features).

This split exists because a single model using `sub_grade`/`int_rate` cannot
legitimately be used to screen a new applicant — those values don't exist
until *after* LendingClub has already priced the loan. See
`notebooks/4_modeling.ipynb` for the full reasoning.

---

## 🎯 Problem Statement

Given loan and borrower characteristics, predict:
- **Default (1)** — loan ends in Charged Off or Default
- **No Default (0)** — loan is Fully Paid

Binary classification on an imbalanced target (~20% default rate among
resolved loans).

---

## 🧠 Methodology

- Full raw-data audit (missingness, cardinality, leakage-risk columns,
  known sentinel values) — see `notebooks/1_data_understanding.ipynb`
- EDA with statistical testing (chi-square) and a time-trend analysis, not
  just visual inspection — see `notebooks/2_eda.ipynb`
- Leakage-safe preprocessing: post-outcome columns excluded at load time,
  not filtered after the fact — see `notebooks/3_data_preprocessing.ipynb`
- **Chronological (out-of-time) train/test split** — not a random split,
  since this data spans multiple years of different economic conditions
- Model selection by an explicit rule (simplest model unless a more complex
  one beats it by a stated margin, backed by a paired t-test), not assumed
- Calibration check (Brier score, reliability curve) alongside ranking
  metrics (ROC-AUC)
- Business-driven decision threshold (minimizes expected cost) rather than
  a default 0.5 cutoff
- Permutation importance alongside impurity-based feature importance
- Baseline comparison against a majority-class model and against
  LendingClub's own `sub_grade` alone, to quantify real model lift

See `notebooks/4_modeling.ipynb` for full detail and reasoning behind each
choice.

---

## 📊 Results

*(Fill in after running `python -m src.train` — values are written to
`models/model_metrics.json`.)*

| Metric | Value |
|---|---|
| Selected model type | `[fill in from model_metrics.json -> chosen_model_type]` |
| Investor Model — Test ROC-AUC | `[fill in -> investor_model.test_roc_auc]` |
| Underwriting Model — Test ROC-AUC | `[fill in -> underwriting_model.test_roc_auc]` |
| sub_grade-alone baseline — AUC | `[fill in -> baselines.subgrade_alone_auc]` |
| Majority-class baseline — Accuracy | `[fill in -> baselines.majority_class_accuracy]` |
| Investor Model — Brier score (calibration) | `[fill in -> investor_model.brier_score]` |
| Optimal decision threshold | `[fill in -> investor_model.optimal_threshold]` |

**Key finding:** `[fill in — e.g. default rate by grade ranges from ~6%
(Grade A) to ~50% (Grade G); see notebooks/5_business_insights.ipynb for
the full expected-loss breakdown by segment]`

---

## 🏗️ Project Structure

```
credit_risk_project/
│
├── app/
│   └── app.py
│
├── dashboard/
│   └── credit_risk_dashboard.pbix
│
├── data/
│   ├── raw/
│   │   └── dataset.csv          (not included -- see Dataset section)
│   ├── processed/
│   │   ├── final_model_data.csv
│   │   ├── dashboard_data.csv
│   │   ├── dashboard_risk_by_grade.csv
│   │   ├── dashboard_risk_by_purpose.csv
│   │   ├── dashboard_risk_by_term.csv
│   │   └── dashboard_risk_by_income.csv
│   └── metadata/
│       └── column_summary.csv
│
├── models/
│   ├── credit_risk_model_investor.joblib
│   ├── expected_features_investor.joblib
│   ├── credit_risk_model_underwriting.joblib
│   ├── expected_features_underwriting.joblib
│   ├── feature_caps.joblib
│   └── model_metrics.json
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_data_preprocessing.ipynb
│   ├── 4_modeling.ipynb
│   └── 5_business_insights.ipynb
│
├── reports/
│   └── credit_risk_dashboard.pdf
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
└── sample_input.csv
```

---

## 📊 Dataset

- **Source:** LendingClub loan data, historical public release (Kaggle mirror)
- **Type:** Tabular financial data, ~2.26M rows, 151 raw columns
- **Target:** Loan outcome (`loan_status`), restricted to resolved loans
  (Fully Paid / Charged Off / Default) for modeling

⚠️ **Note:** The full raw dataset is **not included** in this repository due
to size (2.26M rows).

### To use the full dataset
1. Download the dataset from Kaggle.
2. Place it inside `data/raw/` as `dataset.csv`.

---

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.13 (matches the environment this project was developed
and tested in).

---

## 🚀 How to Run

### 1. Train both models
```bash
python -m src.train
```
Reads `data/raw/dataset.csv`, trains the Investor and Underwriting models,
and saves both plus `model_metrics.json` to `models/`.

### 2. Score new data from the command line
```bash
python -m src.predict path/to/input.csv --model underwriting
```
Use `--model investor` only if `sub_grade` and `int_rate` are already known
for the records being scored.

### 3. Run the notebooks (in order)
```
notebooks/1_data_understanding.ipynb
notebooks/2_eda.ipynb
notebooks/3_data_preprocessing.ipynb
notebooks/4_modeling.ipynb
notebooks/5_business_insights.ipynb
```
Each depends on the previous notebook's output — run in order, top to
bottom. Notebook 5 also produces the CSVs the Power BI dashboard reads from.

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Open the dashboard
`dashboard/credit_risk_dashboard.pbix` in Power BI Desktop — reads from the
CSVs produced by notebook 5 and from `models/model_metrics.json`.

---

## ⚠️ Limitations

- Trained only on LendingClub's **approved** loan population — applicants
  rejected outright are not represented, so the model cannot speak to risk
  among rejected applicants. This is a selection-bias limitation inherent
  to the data source, not fixable within this project.
- The Investor Model's use of `sub_grade`/`int_rate` means a meaningful
  share of its predictive power reflects LendingClub's own risk pricing
  rather than net-new signal — quantified directly in
  `notebooks/5_business_insights.ipynb` (Investor vs Underwriting AUC gap).
- The decision-threshold cost assumptions (loss-given-default, lost margin
  percentages) are illustrative, not verified institutional figures — see
  `config.yaml` under `cost_assumptions`. Replace with real figures before
  using this for an actual lending or investment decision.
- The chronological train/test split approximates real-world deployment
  conditions but is not a substitute for live monitoring once deployed.

---

## 🔭 Future Work

- Live monitoring for model drift once deployed, particularly given the
  chronological-split finding that risk conditions shift over time
- Verified institutional cost figures to replace the illustrative
  threshold-optimization assumptions
- Expand the Power BI dashboard's time-trend view (loan volume and default
  rate by issue year) — supported by the data, not yet on every dashboard
  page