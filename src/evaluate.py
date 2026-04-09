import json
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(y_true, y_pred, y_proba, model_name="baseline"):
    """
    Evaluates model and computes standard metrics.
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "classification_report": report
    }
    return metrics

def save_metrics(metrics, path="models/model_metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
