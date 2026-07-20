"""
Model evaluation utilities for the credit risk project.
"""

import json
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve


def evaluate_model(y_true, y_pred, y_proba, model_name="baseline") -> dict:
    """
    Computes standard classification metrics plus calibration -- calibration
    was entirely absent from the original evaluation and matters directly
    for any expected-loss calculation downstream (notebooks/5_business_insights.ipynb).
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    brier = brier_score_loss(y_true, y_proba)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    metrics = {
        "model_name": model_name,
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "classification_report": report,
        "calibration_curve": {
            "observed": prob_true.tolist(),
            "predicted": prob_pred.tolist(),
        },
    }
    return metrics


def find_optimal_threshold(
    y_true, y_proba, avg_loan_amount: float,
    loss_given_default_pct: float = 0.60,
    lost_margin_pct: float = 0.13,
    thresholds=None,
) -> dict:
    """
    Finds the decision threshold that minimizes total expected cost, rather
    than defaulting to sklearn's implicit 0.5 cutoff.

    IMPORTANT: loss_given_default_pct and lost_margin_pct are illustrative
    assumptions (see config.yaml `cost_assumptions` and
    notebooks/4_modeling.ipynb, Section 10) -- replace with verified
    institutional figures before using this for a real decision.

    Returns
    -------
    dict with the optimal threshold, its expected cost, and the full
    threshold/cost curve for plotting.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    cost_fn = avg_loan_amount * loss_given_default_pct
    cost_fp = avg_loan_amount * lost_margin_pct

    costs = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            preds,
            labels=[0, 1]
        ).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))

    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "optimal_expected_cost": float(costs[best_idx]),
        "thresholds": thresholds.tolist(),
        "expected_costs": costs.tolist(),
        "assumptions": {
            "loss_given_default_pct": loss_given_default_pct,
            "lost_margin_pct": lost_margin_pct,
        },
    }


def save_metrics(metrics: dict, path: str = "models/model_metrics.json") -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, default=str)
