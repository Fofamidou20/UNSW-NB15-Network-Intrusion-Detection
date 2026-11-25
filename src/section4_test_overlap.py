
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.data_loader import load_data


def evaluate_subset(model, X, y):
    if len(X) == 0:
        return None

    predictions = model.predict(X)
    attack_idx = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[
        :, attack_idx
    ]

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Observations": int(len(y)),
        "Attack Percentage": float(100 * y.mean()),
        "Accuracy": float(accuracy_score(y, predictions)),
        "Precision": float(precision_score(
            y, predictions, zero_division=0
        )),
        "Recall": float(recall_score(
            y, predictions, zero_division=0
        )),
        "F1 Score": float(f1_score(
            y, predictions, zero_division=0
        )),
        "ROC-AUC": (
            float(roc_auc_score(y, probabilities))
            if y.nunique() == 2 else None
        ),
        "Average Precision": (
            float(average_precision_score(y, probabilities))
            if y.nunique() == 2 else None
        ),
        "False Positive Rate": (
            float(fp / (fp + tn)) if fp + tn else None
        ),
        "False Negative Rate": (
            float(fn / (fn + tp)) if fn + tp else None
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    return metrics


def main():
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"

    output_dir = (
        results_dir / "section4_test_overlap"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_data()

    ignored = ["id", "label", "attack_cat"]

    X_train = train.drop(
        columns=ignored, errors="ignore"
    )

    X_test = test.drop(
        columns=ignored, errors="ignore"
    )

    X_test = X_test[X_train.columns]
    y_test = test["label"].astype(int)

    # Exact predictor comparison, excluding labels and IDs.
    train_keys = pd.MultiIndex.from_frame(X_train)
    test_keys = pd.MultiIndex.from_frame(X_test)

    overlap_mask = np.asarray(
        test_keys.isin(train_keys)
    )

    print("\nOFFICIAL-TEST OVERLAP ANALYSIS")
    print("=" * 55)
    print("Total test rows:", len(X_test))
    print("Overlapping test rows:", overlap_mask.sum())
    print("Non-overlapping test rows:",
          (~overlap_mask).sum())

    model_path = (
        results_dir / "cv_optimized_random_forest.joblib"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    model = joblib.load(model_path)

    subsets = {
        "Full Official Test": np.ones(
            len(X_test), dtype=bool
        ),
        "Overlapping Test": overlap_mask,
        "Non-Overlapping Test": ~overlap_mask
    }

    records = []

    for name, mask in subsets.items():
        X_subset = X_test.loc[mask]
        y_subset = y_test.loc[mask]

        metrics = evaluate_subset(
            model, X_subset, y_subset
        )

        if metrics is None:
            continue

        records.append({
            "Subset": name,
            **metrics
        })

        print(f"\n{name}")
        print("-" * 40)

        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    results_df = pd.DataFrame(records)

    results_df.to_csv(
        output_dir / "test_overlap_results.csv",
        index=False
    )

    with open(
        output_dir / "test_overlap_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "model_file": model_path.name,
                "overlap_definition":
                    "Exact predictor match against original training CSV",
                "total_test_rows": int(len(X_test)),
                "overlapping_rows": int(overlap_mask.sum()),
                "non_overlapping_rows": int(
                    (~overlap_mask).sum()
                ),
                "results": records
            },
            file,
            indent=4
        )

    print("\nOverlap sensitivity analysis completed!")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
