"""
main.py

Ties the separated modules together into a single runnable pipeline.

Notice that this file is almost entirely *orchestration* -- it doesn't
load data, build models, or compute metrics itself. It just calls the
functions that do, in order. That's the payoff of splitting things into
modules: this file reads like a table of contents for the whole project.
"""

from __future__ import annotations

from iris_classifier.data import prepare_dataset
from iris_classifier.evaluate import evaluate_predictions
from iris_classifier.model import build_model, train_model
from iris_classifier.predict import predict_species


def run() -> None:
    # 1. Data prep
    dataset = prepare_dataset()

    # 2. Model building
    model = build_model(max_depth=3)
    model = train_model(model, dataset.X_train, dataset.y_train)

    # 3. Evaluation
    y_pred = model.predict(dataset.X_test)
    result = evaluate_predictions(dataset.y_test, y_pred, dataset.target_names)

    print("=== Evaluation on held-out test set ===")
    print(f"Accuracy: {result.accuracy:.3f}")
    print()
    print("Confusion matrix:")
    print(result.confusion)
    print()
    print("Classification report:")
    print(result.report)

    # 4. "Deployment" stand-in: predict on one new, made-up sample
    print("=== Predicting on a new sample ===")
    species = predict_species(
        model,
        dataset.target_names,
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
    )
    print(f"Predicted species: {species}")


if __name__ == "__main__":
    run()
