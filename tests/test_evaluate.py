"""Tests for iris_classifier.evaluate

Notice these tests never call sklearn's classifier or load any data --
they hand-craft y_true/y_pred arrays. That's only possible because
evaluate.py was written to depend only on prediction arrays, not on
the model or dataset objects directly.
"""

import numpy as np

from iris_classifier.evaluate import evaluate_predictions


def test_perfect_predictions_give_accuracy_one():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    result = evaluate_predictions(y_true, y_pred)

    assert result.accuracy == 1.0


def test_all_wrong_predictions_give_accuracy_zero():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    result = evaluate_predictions(y_true, y_pred)

    assert result.accuracy == 0.0


def test_confusion_matrix_shape_matches_class_count():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    result = evaluate_predictions(y_true, y_pred)

    # 3 classes -> 3x3 confusion matrix
    assert result.confusion.shape == (3, 3)


def test_report_contains_target_names():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = evaluate_predictions(y_true, y_pred, target_names=["setosa", "versicolor"])

    assert "setosa" in result.report
    assert "versicolor" in result.report
