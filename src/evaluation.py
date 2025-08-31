# src/evaluation.py
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred, labels=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "total_samples": int(len(y_true)),
    }
