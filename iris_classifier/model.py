"""
model.py

Responsible for building and training the classifier.

Keeping model construction separate from data loading and evaluation
means you can swap the algorithm (e.g. DecisionTree -> LogisticRegression)
without touching any other file, and you can unit test "does training
work" without needing to know how the data was prepared.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def build_model(max_depth: int | None = 3, random_state: int = 42) -> DecisionTreeClassifier:
    """Construct an untrained classifier.

    A shallow decision tree is used deliberately: it is easy to reason
    about and visualize, which makes it a good teaching model.
    """
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


def train_model(
    model: DecisionTreeClassifier, X_train: np.ndarray, y_train: np.ndarray
) -> DecisionTreeClassifier:
    """Fit a classifier on training data and return it.

    Returning the (mutated) model keeps the function signature explicit
    about what's happening, and makes it trivial to test: pass in data,
    check the model is fitted afterwards.
    """
    model.fit(X_train, y_train)
    return model
