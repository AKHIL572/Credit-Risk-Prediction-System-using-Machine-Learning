"""
Training script for the credit risk project.

Mirrors notebooks/4_modeling.ipynb exactly -- if you change the modeling
approach in one, change it in the other, or run the risk of the "two
disconnected implementations" problem this project had before (notebooks
and src/ reimplementing the same logic independently and silently
diverging).

Trains TWO models for two distinct, explicitly named personas:
- Investor Risk Model: full features (incl. sub_grade, int_rate) -- for
  scoring already-listed loans.
- Underwriting Screening Model: borrower-only features -- for screening a
  new applicant who has not yet been graded/priced by LendingClub.
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

from src.data_loader import load_lendingclub_data
from src.feature_engineering import create_target, engineer_features
from src.preprocessing import get_feature_types, build_preprocessor
from src.evaluate import evaluate_model, find_optimal_threshold, save_metrics

# =========================
# 1. Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = os.path.join(PROJECT_ROOT, config["paths"]["data"]["raw"])
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = config["training"]["random_state"]
CV_SPLITS = config["training"].get("cv_splits", 3)
IMPROVEMENT_THRESHOLD = config["training"].get("improvement_threshold", 0.01)
LGD_PCT = config["cost_assumptions"]["loss_given_default_pct"]
MARGIN_PCT = config["cost_assumptions"]["lost_margin_pct"]

# issue_d is required for the chronological split -- NOT a model feature.
RAW_REQUIRED_COLS = [
    "loan_status", "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "application_type", "issue_d",
]

INVESTOR_FEATURES = [
    "loan_amnt", "loan_term_numeric", "int_rate", "installment",
    "sub_grade",  # grade dropped -- sub_grade subsumes it
    "emp_length_numeric", "home_ownership",
    "annual_inc_capped", "verification_status", "purpose",
    "dti_capped", "delinq_2yrs", "fico_avg",
    "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "application_type",
    "loan_to_income", "installment_to_income",
]
UNDERWRITING_FEATURES = [
    f for f in INVESTOR_FEATURES if f not in ("sub_grade", "int_rate")]


def main():
    # =========================
    # 2. Load, engineer, split chronologically
    # =========================
    print("Loading data...")
    df = load_lendingclub_data(DATA_PATH, required_cols=RAW_REQUIRED_COLS)
    df = create_target(df)
    # training call -- computes caps fresh
    df = engineer_features(df, caps=None)
    caps = df.attrs["caps"]

    df["issue_d_parsed"] = pd.to_datetime(
        df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.sort_values("issue_d_parsed").reset_index(drop=True)

    split_idx = int(len(df) * (1 - config["training"]["test_size"]))
    split_date = df.loc[split_idx, "issue_d_parsed"]
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    y_train, y_test = train_df["target"], test_df["target"]

    print(f"Split date: {split_date}")
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # =========================
    # 3. Baselines
    # =========================
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[["loan_amnt"]], y_train)
    majority_baseline_acc = round(
        dummy.score(test_df[["loan_amnt"]], y_test), 4)

    subgrade_rank = train_df.groupby("sub_grade")["target"].mean()
    test_subgrade_score = test_df["sub_grade"].map(subgrade_rank)
    valid = test_subgrade_score.notna()
    subgrade_baseline_auc = round(
        roc_auc_score(y_test[valid], test_subgrade_score[valid]), 4
    )
    print(f"Majority-class baseline accuracy: {majority_baseline_acc}")
    print(f"sub_grade-alone baseline ROC-AUC: {subgrade_baseline_auc}")

    # =========================
    # 4. Investor Model -- comparison, selection, tuning
    # =========================
    X_train_inv = train_df[INVESTOR_FEATURES]
    X_test_inv = test_df[INVESTOR_FEATURES]
    num_cols, cat_cols = get_feature_types(X_train_inv)
    preprocessor_inv = build_preprocessor(num_cols, cat_cols)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    simplicity_order = ["Logistic Regression",
                        "Decision Tree", "Random Forest"]
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    cv_results = {}
    for name in simplicity_order:
        pipe = Pipeline([("preprocessing", preprocessor_inv),
                        ("classifier", candidate_models[name])])
        scores = cross_val_score(
            pipe, X_train_inv, y_train, scoring="roc_auc", cv=tscv, n_jobs=-1)
        cv_results[name] = scores
        print(f"{name}: mean ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ==========================================================
    # # Explicit model-selection rule
    # #
    # # A more complex model is only selected if:
    # # 1. It improves mean CV ROC-AUC by more than IMPROVEMENT_THRESHOLD.
    # # 2. The improvement is statistically significant (paired t-test, p < 0.05).
    # #
    # # Otherwise, keep the simplest model.
    # # ==========================================================
    chosen_name = simplicity_order[0]
    baseline_name = simplicity_order[0]
    baseline_scores = cv_results[baseline_name]
    
    print("\nModel selection comparisons:")
    for name in simplicity_order[1:]:
        candidate_scores = cv_results[name]
        
        improvement = (
            candidate_scores.mean() -
            baseline_scores.mean()
            )
        
        t_stat, p_val = stats.ttest_rel(
            candidate_scores,
            baseline_scores
            )
            
        print(
            f"{name} vs {baseline_name}: "
            f"+{improvement:.4f} AUC | "
            f"p = {p_val:.4f}"
        )
        
        if (
            improvement > IMPROVEMENT_THRESHOLD
            and p_val < 0.05
        ):
            chosen_name = name
            baseline_name = name
            baseline_scores = candidate_scores
            
    print(
        f"\nSelected model: {chosen_name} "
        f"(required > {IMPROVEMENT_THRESHOLD:.4f} mean AUC improvement "
        f"AND p < 0.05)"
    )

    param_grids = {
        "Logistic Regression": {"classifier__C": [0.01, 0.1, 1, 10]},
        "Decision Tree": {"classifier__max_depth": [4, 6, 8], "classifier__min_samples_split": [2, 10, 50]},
        "Random Forest": {"classifier__n_estimators": [100, 150], "classifier__max_depth": [6, 8],
                          "classifier__min_samples_split": [2, 5]},
    }
    pipeline = Pipeline([("preprocessing", preprocessor_inv),
                        ("classifier", candidate_models[chosen_name])])
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grids[chosen_name], n_iter=4,
        scoring="roc_auc", cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train_inv, y_train)
    investor_model = search.best_estimator_
    print("Best params:", search.best_params_)

    # =========================
    # 5. Investor Model -- evaluation, calibration, threshold, importance
    # =========================
    train_proba = investor_model.predict_proba(X_train_inv)[:, 1]
    test_proba = investor_model.predict_proba(X_test_inv)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    print(f"Investor Model -- Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} "
          f"| Gap: {train_auc - test_auc:.4f}")

    investor_metrics = evaluate_model(
        y_test, (test_proba >= 0.5).astype(int), test_proba, "Investor Model")

    avg_loan_amount = float(train_df["loan_amnt"].mean())
    threshold_result = find_optimal_threshold(
        y_test, test_proba, avg_loan_amount,
        loss_given_default_pct=LGD_PCT, lost_margin_pct=MARGIN_PCT
    )
    print(
        f"Optimal decision threshold: {threshold_result['optimal_threshold']}")

    perm_result = permutation_importance(
        investor_model, X_test_inv, y_test, n_repeats=5,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
    )
    perm_importance = pd.Series(
        perm_result.importances_mean, index=INVESTOR_FEATURES
    ).sort_values(ascending=False)
    print("Top permutation importances:\n", perm_importance.head(10))

    # =========================
    # 6. Underwriting Model -- same model type/params, borrower-only features
    # =========================
    X_train_uw = train_df[UNDERWRITING_FEATURES]
    X_test_uw = test_df[UNDERWRITING_FEATURES]
    num_cols_uw, cat_cols_uw = get_feature_types(X_train_uw)
    preprocessor_uw = build_preprocessor(num_cols_uw, cat_cols_uw)

    underwriting_model = Pipeline([
        ("preprocessing", preprocessor_uw),
        ("classifier", candidate_models[chosen_name]),
    ])
    underwriting_model.set_params(**{
        k: v for k, v in search.best_params_.items() if k in underwriting_model.get_params()
    })
    underwriting_model.fit(X_train_uw, y_train)
    uw_test_proba = underwriting_model.predict_proba(X_test_uw)[:, 1]
    uw_test_auc = roc_auc_score(y_test, uw_test_proba)
    print(f"Underwriting Model Test AUC: {uw_test_auc:.4f} "
          f"(gap vs Investor Model attributable to sub_grade/int_rate: {test_auc - uw_test_auc:.4f})")

    # =========================
    # 7. Save everything
    # =========================
    joblib.dump(investor_model, os.path.join(
        MODEL_DIR, "credit_risk_model_investor.joblib"))
    joblib.dump(INVESTOR_FEATURES, os.path.join(
        MODEL_DIR, "expected_features_investor.joblib"))
    joblib.dump(underwriting_model, os.path.join(
        MODEL_DIR, "credit_risk_model_underwriting.joblib"))
    joblib.dump(UNDERWRITING_FEATURES, os.path.join(
        MODEL_DIR, "expected_features_underwriting.joblib"))
    joblib.dump(caps, os.path.join(MODEL_DIR, "feature_caps.joblib"))

    metrics = {
        "chosen_model_type": chosen_name,
        "best_params": search.best_params_,
        "split_date": str(split_date),
        "baselines": {
            "majority_class_accuracy": majority_baseline_acc,
            "subgrade_alone_auc": subgrade_baseline_auc,
        },
        "investor_model": {
            "train_roc_auc": float(train_auc),
            "test_roc_auc": float(test_auc),
            "brier_score": investor_metrics["brier_score"],
            "optimal_threshold": threshold_result["optimal_threshold"],
            "threshold_cost_assumptions": threshold_result["assumptions"],
            "top_permutation_importances": perm_importance.head(10).to_dict(),
        },
        "underwriting_model": {
            "test_roc_auc": float(uw_test_auc),
        },
    }
    save_metrics(metrics, os.path.join(MODEL_DIR, "model_metrics.json"))
    print(f"\nSaved both models, feature caps, and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
