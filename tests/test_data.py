"""Tests for iris_classifier.data

These tests check data loading and splitting in isolation -- they never
touch the model or evaluation code. That's the point of separating
modules: this file can fail or pass independently of whether the
classifier itself works.
"""

from iris_classifier.data import load_raw_iris, prepare_dataset


def test_load_raw_iris_has_expected_shape():
    raw = load_raw_iris()
    # 150 samples, 4 features, as documented for the classic Iris dataset
    assert raw.data.shape == (150, 4)
    assert raw.target.shape == (150,)


def test_load_raw_iris_has_three_classes():
    raw = load_raw_iris()
    assert len(set(raw.target)) == 3


def test_prepare_dataset_splits_correctly():
    dataset = prepare_dataset(test_size=0.2, random_state=42)

    total = len(dataset.X_train) + len(dataset.X_test)
    assert total == 150

    # roughly 20% in the test set (allow rounding)
    assert 25 <= len(dataset.X_test) <= 35


def test_prepare_dataset_feature_and_target_names():
    dataset = prepare_dataset()
    assert len(dataset.feature_names) == 4
    assert len(dataset.target_names) == 3


def test_prepare_dataset_is_reproducible():
    first = prepare_dataset(random_state=123)
    second = prepare_dataset(random_state=123)

    assert (first.X_train == second.X_train).all()
    assert (first.y_train == second.y_train).all()
