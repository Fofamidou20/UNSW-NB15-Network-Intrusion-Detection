
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "section6_verification"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "neural_network": "neural_network.joblib",
    "optimized_random_forest": "optimized_random_forest.joblib",
    "cv_optimized_random_forest":
        "cv_optimized_random_forest.joblib",
}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()

def evaluate(model, X, y):
    predictions = model.predict(X)

    classes = list(model.classes_)
    positive_index = classes.index(1)

    probabilities = model.predict_proba(X)[
        :, positive_index
    ]

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
    }, predictions

def main():
    train_path = (
        DATA / "UNSW_NB15_training-set.csv"
    )
    test_path = (
        DATA / "UNSW_NB15_testing-set.csv"
    )

    print("\nDATASET CHECKSUMS")
    print("Training:", sha256(train_path))
    print("Testing:", sha256(test_path))

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    ignored = ["id", "label", "attack_cat"]

    X_test = test.drop(
        columns=ignored, errors="ignore"
    )
    y_test = test["label"].astype(int)

    print("\nDATASET INFORMATION")
    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Test features:", X_test.shape[1])

    records = []
    predictions_by_model = {}

    for name, filename in MODEL_FILES.items():
        path = RESULTS / filename

        print("\n" + "=" * 60)
        print("MODEL:", name)

        if not path.exists():
            print("MISSING:", path)
            continue

        print("Artifact SHA256:", sha256(path))

        # Only load trusted artifacts created locally.
        model = joblib.load(path)

        classifier = (
            model.named_steps["classifier"]
            if hasattr(model, "named_steps")
            else model
        )

        print("Classifier:", type(classifier).__name__)
        print("Parameters:", classifier.get_params())

        metrics, predictions = evaluate(
            model, X_test, y_test
        )

        predictions_by_model[name] = predictions

        record = {
            "model": name,
            "artifact": filename,
            "artifact_sha256": sha256(path),
            **metrics,
        }
        records.append(record)

        print("\nRecomputed metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    comparison = pd.DataFrame(records)

    comparison.to_csv(
        OUTPUT / "verified_model_metrics.csv",
        index=False
    )

    with open(
        OUTPUT / "verified_model_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(records, file, indent=4)

    a = "optimized_random_forest"
    b = "cv_optimized_random_forest"

    if (
        a in predictions_by_model
        and b in predictions_by_model
    ):
        different = (
            predictions_by_model[a]
            != predictions_by_model[b]
        ).sum()

        print("\nOPTIMIZED MODEL COMPARISON")
        print("Different predictions:", int(different))
        print(
            "Prediction disagreement rate:",
            different / len(y_test)
        )

    print("\nVERIFICATION COMPLETED")
    print("Results saved to:", OUTPUT)

if __name__ == "__main__":
    main()
