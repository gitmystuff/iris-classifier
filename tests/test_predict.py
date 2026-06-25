"""Tests for iris_classifier.predict

These tests train a tiny throwaway model rather than importing the
full data/model pipeline, keeping this test file fast and focused
purely on whether predict_species wires things together correctly.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from iris_classifier.predict import predict_species


def _tiny_trained_model():
    model = DecisionTreeClassifier(max_depth=2, random_state=0)
    X_train = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [9, 9, 9, 9], [9, 9, 9, 9]])
    y_train = np.array([0, 0, 1, 1])
    model.fit(X_train, y_train)
    return model


def test_predict_species_returns_a_known_name():
    model = _tiny_trained_model()
    target_names = ["small_flower", "big_flower"]

    species = predict_species(
        model,
        target_names,
        sepal_length=1,
        sepal_width=1,
        petal_length=1,
        petal_width=1,
    )

    assert species in target_names


def test_predict_species_picks_the_closer_class():
    model = _tiny_trained_model()
    target_names = ["small_flower", "big_flower"]

    species = predict_species(
        model,
        target_names,
        sepal_length=9,
        sepal_width=9,
        petal_length=9,
        petal_width=9,
    )

    assert species == "big_flower"
