from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.7,
    candidate_thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5
    best_metrics = evaluate_binary_classifier(y_true, y_proba, threshold=0.5)

    for threshold in candidate_thresholds:
        metrics = evaluate_binary_classifier(y_true, y_proba, threshold=float(threshold))
        recall_ok = metrics["recall"] >= min_recall
        if not recall_ok:
            continue
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
