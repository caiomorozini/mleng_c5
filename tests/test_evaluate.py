import numpy as np

from src.evaluate import evaluate_binary_classifier, find_best_threshold


def test_evaluate_binary_classifier_returns_main_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.6, 0.4, 0.7])

    metrics = evaluate_binary_classifier(y_true, y_proba, threshold=0.5)

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert metrics["recall"] > 0


def test_find_best_threshold_honors_min_recall():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.4, 0.7, 0.2])

    threshold, metrics = find_best_threshold(y_true, y_proba, min_recall=0.6)

    assert 0.1 <= threshold <= 0.9
    assert metrics["recall"] >= 0.6
