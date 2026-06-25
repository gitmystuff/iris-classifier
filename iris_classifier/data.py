"""
data.py

Responsible for loading and preparing the Iris dataset.

This module isolates all "data prep" concerns so that the rest of the
codebase (model building, evaluation) never has to know where the data
came from or how it was split. That separation is what makes each piece
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@dataclass
class IrisDataset:
    """A simple container for the split Iris data and its metadata."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_raw_iris():
    """Load the raw Iris dataset as a scikit-learn Bunch object.

    Kept as its own tiny function so tests (and students) can inspect
    the unmodified dataset before any splitting happens.
    """
    return load_iris()


def prepare_dataset(test_size: float = 0.2, random_state: int = 42) -> IrisDataset:
    """Load the Iris dataset and split it into train/test sets.

    Parameters
    ----------
    test_size : float
        Fraction of the data to hold out for testing.
    random_state : int
        Seed for reproducible splits.

    Returns
    -------
    IrisDataset
        A dataclass bundling the split arrays and label metadata.
    """
    raw = load_raw_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        raw.data,
        raw.target,
        test_size=test_size,
        random_state=random_state,
        stratify=raw.target,
    )

    return IrisDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
    )
