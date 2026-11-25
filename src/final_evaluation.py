
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_loader import load_data


def run_final_evaluation():

    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the UNSW-NB15 testing dataset
    _, test = load_data()

    X_test = test.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )
    y_test = test["label"].astype(int)

    # Load cross-validation-selected model
    model_path = (
        results_dir / "cv_optimized_random_forest.joblib"
    )

    model = joblib.load(model_path)

    print("Evaluating cross-validation-selected Random Forest...")

    # Generate predictions
    predictions = model.predict(X_test)

    # Obtain probability estimates for attack class
    attack_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, attack_index]

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Average Precision": average_precision_score(
            y_test, probabilities
        ),
        "False Positive Rate": (
            fp / (fp + tn) if fp + tn else 0
        ),
        "False Negative Rate": (
            fn / (fn + tp) if fn + tp else 0
        ),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp)
    }

    # Save results
    pd.DataFrame([metrics]).to_csv(
        results_dir / "final_model_evaluation.csv",
        index=False
    )

    with open(
        results_dir / "final_model_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Generate confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Normal", "Attack"],
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Cross-Validation-Selected Random Forest Confusion Matrix"
    )
    plt.tight_layout()

    plt.savefig(
        results_dir / "final_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nFinal Model Evaluation Results:")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nFinal model evaluation completed!")


if __name__ == "__main__":
    run_final_evaluation()
