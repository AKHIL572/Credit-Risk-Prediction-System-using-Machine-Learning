# 💳 Credit Risk Prediction System

A complete end-to-end **Machine Learning project** that predicts the probability of loan default using historical loan data.  
This project follows **real-world industry practices**, from data preprocessing and model training to deployment using **Streamlit**.

---

## 📌 Project Overview

Financial institutions face significant risk when issuing loans. Incorrect decisions can lead to high default rates and financial losses.

This project aims to:
- Predict whether a customer is likely to **default on a loan**
- Provide **probability-based risk scores**
- Help financial teams make **data-driven lending decisions**

The system uses supervised machine learning models trained on historical loan data and is deployed as an interactive web application.

---

## 🎯 Problem Statement

Given customer and loan-related information, predict:
- **Default (1)** – High risk borrower  
- **No Default (0)** – Low risk borrower  

This is a **binary classification problem** with imbalanced classes.

---

## 🧠 Machine Learning Approach

- Data cleaning & preprocessing
- Feature engineering
- Model training and comparison
- Hyperparameter tuning
- Final model selection using ROC-AUC
- Model serialization
- Deployment using Streamlit

---

## 🏗️ Project Structure

```
credit_risk_project/
│
├── Dataset/
│ └── column_summary.csv # original dataset is too large
│
├── Models/
│ ├── credit_risk_model.joblib # Trained ML pipeline
│ └── expected_features.joblib # Feature schema
│
├── Notebooks/
│ ├── 1_data_understanding.ipynb
│ ├── 2_data_preprocessing.ipynb
│ └── 3_preprocessing_&_modelling.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── train.py
│ └── predict.py
│
├── app.py # Streamlit application
├── requirements.txt
├── sample_input.csv # Sample data for testing
└── README.md
```


---

## 📊 Dataset Information

- **Dataset**: Lending Club Loan Data
- **Type**: Tabular financial data
- **Target Variable**: Loan default status

⚠️ **Note**:  
The full dataset is **not included** in this repository due to GitHub size limitations.

### To use the full dataset:
1. Download the dataset from Kaggle
2. Place it inside the `Dataset/` folder
3. Rename it as `dataset.csv`

---

## ⚙️ Models Used

- Logistic Regression (baseline)
- Decision Tree Classifier
- Random Forest Classifier (final model)

### Evaluation Metrics:
- ROC-AUC
- Precision
- Recall
- F1-score

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
