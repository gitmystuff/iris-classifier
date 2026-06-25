"""
evaluate.py

Responsible for evaluating a trained model's performance.

This module never trains anything and never touches raw data loading —
it only takes predictions and ground truth in, and metrics out. That
narrow responsibility is exactly what makes it easy to test with fake,
hand-crafted arrays instead of needing the real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvaluationResult:
    """Bundles the metrics produced by evaluating a model's predictions."""

    accuracy: float
    confusion: np.ndarray
    report: str


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None
) -> EvaluationResult:
    """Compute accuracy, a confusion matrix, and a text classification report.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions, same length and order as y_true.
    target_names : list[str], optional
        Human-readable class names for the report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)

    return EvaluationResult(accuracy=accuracy, confusion=confusion, report=report)
