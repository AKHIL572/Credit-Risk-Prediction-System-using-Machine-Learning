# 💳 Credit Risk Prediction System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

An end-to-end **Machine Learning**, **Business Intelligence**, and **Data Analytics** project that predicts the probability of loan default using historical LendingClub loan data.

The project follows a complete real-world machine learning workflow beginning with data understanding and exploratory analysis, progressing through feature engineering and model development, and ending with deployment through a Streamlit web application and interactive Power BI dashboard.

Unlike many credit risk projects that build a single generic classifier, this project develops **two separate machine learning models** for two distinct business use cases:

- **Investor Risk Model** – predicts the default risk of an already-listed LendingClub loan using both borrower information and LendingClub's own pricing signals.
- **Underwriting Screening Model** – predicts default risk for a brand-new loan applicant using only borrower information, making it suitable for pre-approval screening.

---

# 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Project Methodology](#-project-methodology)
- [Installation](#️-installation)
- [Running the Project](#-running-the-project)
- [Model Outputs](#-model-outputs)
- [Results](#-results)
- [Dashboard](#-dashboard)
- [Streamlit Application](#️-streamlit-application)
- [Limitations](#️-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

# 📌 Project Overview

Credit risk assessment is one of the most important problems in retail lending.

Every loan issued by a financial institution carries uncertainty regarding repayment. Incorrectly approving high-risk borrowers results in financial losses, while incorrectly rejecting low-risk borrowers reduces potential revenue.

The objective of this project is to build an interpretable and practical machine learning system capable of estimating the probability that a loan will default before it is fully repaid.

The project was developed using the publicly available **LendingClub Loan Dataset**, containing approximately **2.26 million historical loan records** with over **150 original variables** describing borrower demographics, financial characteristics, loan information, and repayment outcomes.

The project combines:

- Data Analytics
- Statistical Analysis
- Machine Learning
- Business Intelligence
- Model Deployment

into one integrated solution.

---

## 🚀 Project Highlights

- Built two separate credit risk models for distinct business scenarios
- Trained on ~2.26 million LendingClub loan records
- Implemented leakage-safe preprocessing and chronological train/test splitting
- Compared multiple algorithms using TimeSeries cross-validation and statistical model selection
- Optimized decision thresholds using business cost assumptions instead of a fixed 0.5 cutoff
- Developed an interactive Power BI dashboard with expected-loss analysis
- Deployed the models using a Streamlit web application supporting batch and single-applicant predictions

---

# 🎯 Business Problem

Traditional classification models often optimise only statistical metrics such as accuracy or ROC-AUC.

However, in real lending environments, business decisions involve financial trade-offs rather than classification accuracy alone.

For example:

- Approving a borrower who later defaults creates financial losses.
- Rejecting a reliable borrower reduces future profit.
- Different stakeholders require different prediction systems.

This project therefore addresses **two independent business scenarios** rather than forcing one model to solve both.

### 1. Investor Risk Model

This model predicts the probability that an **already-listed LendingClub loan** will default.

Since the loan has already been evaluated by LendingClub, variables such as:

- Sub Grade
- Interest Rate

are already available and can legitimately be used by the model.

Typical user:

- Retail Investor
- Institutional Investor
- Portfolio Manager

### 2. Underwriting Screening Model

This model predicts default risk **before** LendingClub assigns a grade or interest rate.

Only borrower information available at the application stage is used.

Typical user:

- Loan Officer
- Credit Analyst
- Underwriting Team

This separation avoids information leakage while reflecting how predictive models are actually deployed in financial institutions.

---

# 🎯 Project Objectives

The primary objectives of this project are:

- Understand the characteristics of historical LendingClub loans.
- Identify factors associated with loan default.
- Build reliable credit risk prediction models.
- Prevent target leakage throughout the modeling pipeline.
- Compare multiple machine learning algorithms.
- Select the best model using objective statistical criteria.
- Evaluate probability calibration.
- Determine an economically optimal decision threshold.
- Deploy the trained models through a Streamlit application.
- Present business insights through an interactive Power BI dashboard.

---

# ⭐ Key Features

### Data Engineering

- Efficient chunk-wise loading for large datasets
- Missing value analysis
- Outlier treatment
- Sentinel value correction
- Feature transformation
- Leakage prevention
- Training-safe preprocessing pipeline

### Machine Learning

- Binary classification
- Two independent production models
- Time-based train/test split
- TimeSeries cross-validation
- Hyperparameter optimisation
- Statistical model comparison
- Probability calibration
- Threshold optimisation
- Permutation feature importance

### Business Analytics

- Executive KPI dashboard
- Loan portfolio analysis
- Risk segmentation
- Expected loss estimation
- Interactive Power BI reports

### Deployment

- Streamlit web application
- Batch prediction
- Single applicant prediction
- Downloadable prediction reports
- Command-line prediction utility

---

# 🛠 Technology Stack

| Category | Tools |
|---|---|
| Programming | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Microsoft Power BI |
| Web Application | Streamlit |
| Model Persistence | Joblib |
| Configuration | YAML |

---

# 🏗 Project Architecture

```text
                    LendingClub Dataset
                           │
                           ▼
                 Data Understanding
                           │
                           ▼
               Exploratory Data Analysis
                           │
                           ▼
               Data Cleaning & Validation
                           │
                           ▼
                 Feature Engineering
                           │
                           ▼
          Chronological Train/Test Split
                           │
                           ▼
               Model Development Pipeline
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 Investor Risk Model              Underwriting Model
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
                  Model Evaluation
                           │
                           ▼
            Business Threshold Optimisation
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
      Streamlit Application        Power BI Dashboard
```

---

# 📂 Repository Structure

```text
credit_risk_prediction/
│
├── app/
│   └── app.py
│
├── dashboard/
│   └── credit_risk_dashboard.pbix
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_data_preprocessing.ipynb
│   ├── 4_modeling.ipynb
│   └── 5_business_insights.ipynb
│
├── reports/
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── config.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📊 Dataset

**Dataset:** LendingClub Historical Loan Dataset

### Dataset Characteristics

| Property | Value |
|-----------|--------|
| Source | LendingClub Public Dataset (Kaggle Mirror) |
| Records | ~2.26 Million |
| Original Features | 151 |
| Target Variable | loan_status |
| Problem Type | Binary Classification |

Only loans with known repayment outcomes were used for model development.

The target variable was constructed as:

| Loan Status | Target |
|-------------|--------|
| Fully Paid | 0 |
| Charged Off | 1 |
| Default | 1 |

Historical credit-policy exception records were mapped back to their underlying resolved loan status before training.

Because the original dataset exceeds GitHub's file size limits, it is **not included** in this repository.

Download the dataset separately and place it inside:

```text
data/raw/dataset.csv
```

---

# 🧠 Project Methodology

This project follows a complete end-to-end machine learning workflow designed to resemble an industry data science project rather than a simple predictive model.

## 1. Data Understanding

The raw LendingClub dataset contains over **2.26 million loan records** and **151 features**. Before any modeling, a complete data audit was performed to identify:

- Missing values
- Duplicate records
- Data types
- High-cardinality features
- Data quality issues
- Potential data leakage
- Target variable distribution
- Business meaning of every important feature

Special handling was implemented for legacy LendingClub policy exception records and unresolved loan statuses.

## 2. Exploratory Data Analysis (EDA)

A detailed EDA was performed to understand borrower behavior and default patterns.

Analysis included:

- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Correlation Analysis
- Default Rate Analysis
- Loan Grade Analysis
- Income Analysis
- Purpose Analysis
- Home Ownership Analysis
- FICO Score Analysis
- Revolving Credit Analysis
- Verification Status Analysis
- Time Trend Analysis

Where appropriate, statistical significance was validated using Chi-Square tests rather than relying only on visual observations.

## 3. Data Preprocessing

The preprocessing pipeline was built using Scikit-Learn Pipelines to ensure leakage-safe transformations.

Key preprocessing steps include:

- Missing value treatment
- Numerical median imputation
- Categorical mode imputation
- One-Hot Encoding
- Feature Scaling
- Type corrections
- Sentinel value correction
- Outlier treatment
- Winsorization
- Feature engineering
- Train-only preprocessing
- Chronological train/test split

All preprocessing transformations are automatically reused during prediction.

## 4. Feature Engineering

Several business-driven features were created to improve predictive performance.

Examples include:

- Average FICO Score
- Loan Term (Numeric)
- Annual Income (Capped)
- Debt-to-Income Ratio (Capped)
- Employment Length (Ordinal)
- Loan-to-Income Ratio
- Installment-to-Income Ratio

Feature engineering follows the exact same pipeline during both training and inference.

## 5. Model Development

Instead of assuming one algorithm is better, multiple models were compared.

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation was performed using:

- TimeSeries Cross Validation
- ROC-AUC
- Paired t-test
- Model simplicity principle

The simplest model is selected unless a more complex model provides statistically meaningful improvement.

## 6. Model Evaluation

Model performance was evaluated using multiple metrics rather than accuracy alone.

Evaluation includes:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Brier Score
- Calibration Curve
- Baseline Comparisons
- Permutation Feature Importance

Business cost optimization is also performed by identifying the decision threshold that minimizes expected financial loss.

## 7. Business Insights

The trained models are used to score the full resolved loan population — not just describe it historically. This produces:

- Predicted default probability per loan, for both the Investor and Underwriting models
- Data-driven risk tiers (quartile-based, not arbitrary cutoffs)
- Expected loss by grade and by purpose (predicted PD × loan amount × loss-given-default)
- A direct comparison of model lift against LendingClub's own `sub_grade` alone

These outputs feed the Power BI dashboard and are what connect it to the trained models rather than to raw historical aggregation alone.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AKHIL572/credit-risk-prediction.git

cd credit-risk-prediction
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the Dataset

Download the LendingClub dataset and place it inside:

```
data/raw/dataset.csv
```

The dataset is excluded from this repository because of its large size.

---

# 🚀 Running the Project

## Step 1 — Train the Models

```bash
python -m src.train
```

This will:

- Load the dataset
- Perform feature engineering
- Build preprocessing pipelines
- Train both machine learning models
- Evaluate model performance
- Find the optimal decision threshold
- Save trained models, preprocessing artifacts, and evaluation metrics

## Step 2 — Predict Using the Command Line

### Underwriting Model

```bash
python -m src.predict sample_input.csv --model underwriting
```

### Investor Model

```bash
python -m src.predict sample_input.csv --model investor
```

Prediction results are automatically saved in `data/processed/`.

## Step 3 — Launch the Streamlit Application

```bash
streamlit run app/app.py
```

The application provides:

- Batch CSV Prediction
- Single Applicant Prediction
- Model Performance Dashboard
- Interactive Risk Visualization
- CSV Export
- Adjustable Decision Threshold

## Step 4 — Open the Power BI Dashboard

Open `dashboard/credit_risk_dashboard.pbix` using Power BI Desktop.

The dashboard is built from the processed CSV files generated during the business insights stage.

---

# 📊 Results

*(Values below are pulled from `models/model_metrics.json` after running `python -m src.train` — replace the placeholders with your actual run's output before publishing.)*

| Metric | Value |
|---|---|
| Selected model type | Logistic Regression |
| Investor Model — Test ROC-AUC | 0.89 |
| Underwriting Model — Test ROC-AUC | 0.81 |
| Baseline ROC-AUC | 0.74 |

**Default rate by grade** (from `data/processed/dashboard_risk_by_grade.csv`): ranges from approximately **6% (Grade A)** to **50% (Grade G)**.

**Model lift:** The Investor Model achieved a **ROC-AUC of 0.6968**, compared with **0.6877** for the Underwriting Model, representing an improvement of **0.0091 ROC-AUC (approximately 0.91 percentage points)**. This relatively small gap suggests that LendingClub's own `sub_grade` and `int_rate` provide only a modest increase in predictive performance beyond the borrower-level information used by the Underwriting Model.

---

# 📊 Dashboard

The dashboard was developed in Microsoft Power BI using processed datasets generated during the business insights stage. It combines historical loan portfolio analysis with machine learning predictions and expected-loss estimates for interactive business reporting.

Key dashboard capabilities include:

- Executive KPI Cards (total loans, actual default rate, model AUC vs. baseline)
- Default Rate Analysis by grade, purpose, and term
- Actual vs. model-predicted default rate by risk tier
- Total expected loss by grade and purpose
- Income-based risk analysis
- Credit score (FICO) analysis
- Grade / purpose / term / income-bracket filters

The dashboard is intended for business users, investors, and underwriting teams.

---

# 🖥️ Streamlit Application

The Streamlit application provides an interactive interface for both business users and analysts to evaluate credit risk using the trained models.

## Batch Prediction

- Upload CSV files
- Predict thousands of applicants
- Download prediction results
- View probability distributions
- View portfolio summaries

## Single Applicant Prediction

Enter borrower information manually and instantly receive:

- Default Probability
- Risk Category
- Decision Threshold
- Prediction Confidence

## Model Performance

Displays:

- ROC-AUC
- Baseline Comparison
- Calibration Score
- Feature Importance
- Selected Model
- Decision Threshold

---

# ⚠️ Limitations

- The dataset contains only approved LendingClub loans. Rejected loan applications are unavailable, so the model cannot estimate risk for rejected applicants.
- Historical data reflects LendingClub's lending policies during the collection period and may not represent current market conditions.
- The Investor Model relies on `sub_grade` and `int_rate`, which are assigned after LendingClub's internal assessment. Therefore, it should not be used for evaluating brand-new loan applications.
- Expected-loss calculations use illustrative business assumptions. Financial institutions should replace these values with organization-specific estimates before deployment.
- Like all supervised learning models, performance may degrade over time due to changes in borrower behavior or economic conditions, making periodic retraining and monitoring important.

---

# 🔮 Future Improvements

Potential enhancements include:

- Model monitoring and drift detection
- Automated retraining pipeline
- Explainable AI using SHAP values
- Hyperparameter optimization with Optuna
- REST API deployment using FastAPI
- Docker containerization
- Cloud deployment (AWS / Azure / GCP)
- CI/CD integration with GitHub Actions
- Real-time prediction service
- Integration with live loan application systems

---

# 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file in the repository root for full terms.

> If a `LICENSE` file is not yet present in your repository, add one before publishing — a license badge without a corresponding `LICENSE` file is inconsistent. GitHub can generate a standard MIT `LICENSE` file automatically when creating or editing the repository (Add file → Create new file → name it `LICENSE` → "Choose a license template").

---

# 🙏 Acknowledgements

- LendingClub for providing historical loan data.
- Kaggle for hosting the publicly available dataset.
- Scikit-Learn development team.
- Streamlit development team.
- Microsoft Power BI.

---

# 📬 Contact

**Akhil T V**

Data Analyst | Aspiring Data Scientist

- **LinkedIn:** https://www.linkedin.com/in/akhil-t-v/
- **GitHub:** https://github.com/AKHIL572
- **Email:** akhilthottekkat135@gmail.com

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!