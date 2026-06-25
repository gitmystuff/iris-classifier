"""
predict.py

A small, focused module for predicting on a single new sample.

This stands in for the "deployment" step in this course: rather than
shipping the model to a server, we expose one clean function that takes
raw measurements and returns a label. In a real deployment, this exact
function is the kind of logic that would sit behind an API endpoint --
the wrapping (Flask route, container, etc.) would change, but this
function would not need to.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def predict_species(
    model: DecisionTreeClassifier,
    target_names: list[str],
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    """Predict the Iris species for a single flower's measurements.

    Parameters
    ----------
    model : DecisionTreeClassifier
        A trained classifier (see model.py).
    target_names : list[str]
        Class names in the same order the model was trained on.
    sepal_length, sepal_width, petal_length, petal_width : float
        Measurements in centimeters.

    Returns
    -------
    str
        The predicted species name.
    """
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_index = model.predict(sample)[0]
    return target_names[predicted_index]
