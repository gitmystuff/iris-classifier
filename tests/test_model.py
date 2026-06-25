"""Tests for iris_classifier.model

These tests use small, fast, hand-built data rather than the full Iris
dataset where possible -- a unit test for "does training mutate the
model object" shouldn't need to depend on data.py at all.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from iris_classifier.model import build_model, train_model


def test_build_model_returns_decision_tree():
    model = build_model()
    assert isinstance(model, DecisionTreeClassifier)


def test_build_model_respects_max_depth():
    model = build_model(max_depth=2)
    assert model.max_depth == 2


def test_train_model_fits_the_model():
    model = build_model(max_depth=2)

    # tiny, easily-separable fake dataset -- no need for real Iris data
    X_train = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    y_train = np.array([0, 0, 1, 1])

    trained = train_model(model, X_train, y_train)

    # a fitted sklearn classifier exposes classes_; an unfitted one does not
    assert hasattr(trained, "classes_")
    assert list(trained.classes_) == [0, 1]


def test_train_model_can_predict_after_fitting():
    model = build_model(max_depth=2)

    X_train = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    y_train = np.array([0, 0, 1, 1])

    trained = train_model(model, X_train, y_train)
    prediction = trained.predict(np.array([[0, 0]]))

    assert prediction[0] == 0
