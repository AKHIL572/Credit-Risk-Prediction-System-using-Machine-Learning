import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.data_loader import load_lendingclub_data
from src.preprocessing import get_feature_types, build_preprocessor


# =========================
# 1. Paths & Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "expected_features.joblib")

REQUIRED_COLS = [
    "loan_status", "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "application_type"
]


# =========================
# 2. Load data (chunked)
# =========================
df = load_lendingclub_data(
    file_path=DATA_PATH,
    required_cols=REQUIRED_COLS
)


# =========================
# 3. Target creation
# =========================
df["target"] = df["loan_status"].map({
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1
})

X = df.drop(columns=["loan_status", "target"])
y = df["target"]


# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================
# 5. Preprocessing
# =========================
num_cols, cat_cols = get_feature_types(X_train)
preprocessor = build_preprocessor(num_cols, cat_cols)


# =========================
# 6. Baseline model
# =========================
baseline_model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
y_proba = baseline_model.predict_proba(X_test)[:, 1]

print("\nBaseline Logistic Regression")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# =========================
# 7. Model comparison
# =========================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", clf)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    print(f"{name} CV ROC-AUC: {scores.mean():.4f}")


# =========================
# 8. Hyperparameter tuning
# =========================
rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])

param_dist = {
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [6, 8],
    "classifier__min_samples_split": [2, 5]
}

random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=4,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=1,   # memory-safe
    verbose=2
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\nBest Params:", random_search.best_params_)
print("Best CV ROC-AUC:", random_search.best_score_)


# =========================
# 9. Final evaluation
# =========================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nFinal Model Performance")
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))


# =========================
# 10. Save artifacts
# =========================
MODEL_DIR = os.path.join(BASE_DIR, "..", "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "credit_risk_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "expected_features.joblib")

joblib.dump(best_model, MODEL_PATH)
joblib.dump(X_train.columns.tolist(), FEATURES_PATH)

print(f"Model saved to {MODEL_PATH}")
