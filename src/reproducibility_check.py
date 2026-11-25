
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.data_loader import load_data


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_reproducibility():

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"

    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 40)

    # Verify datasets
    for filename in [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv"
    ]:
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(path)

        print(f"{filename}: {sha256_file(path)}")

    # Load dataset
    _, test = load_data()

    X_test = test.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )
    y_test = test["label"].astype(int)

    # Load saved CV-selected model
    model = joblib.load(
        results_dir / "cv_optimized_random_forest.joblib"
    )

    predictions = model.predict(X_test)
    attack_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        X_test
    )[:, attack_index]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    recalculated = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp)
    }

    # Compare with saved evaluation
    saved = pd.read_csv(
        results_dir / "final_model_evaluation.csv"
    ).iloc[0]

    for metric, value in recalculated.items():

        if metric not in saved.index:
            raise KeyError(f"Missing saved metric: {metric}")

        difference = abs(float(saved[metric]) - float(value))

        if difference > 1e-8:
            raise AssertionError(
                f"{metric} mismatch: {difference}"
            )

        print(f"PASS: {metric}")

    summary = {
        "status": "passed",
        "sklearn_version": sklearn.__version__,
        "test_records": len(test),
        "metrics": recalculated
    }

    with open(
        results_dir / "reproducibility_audit.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nAll saved-result consistency checks passed!")


if __name__ == "__main__":
    verify_reproducibility()
