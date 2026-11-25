
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
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
from src.preprocessing import build_preprocessor


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Average Precision": average_precision_score(
            y_true, y_prob
        ),
        "False Positive Rate": fp / (fp + tn),
        "False Negative Rate": fn / (fn + tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def main():
    project_dir = Path(__file__).resolve().parent.parent
    output_dir = (
        project_dir / "results" / "section4_group_cv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train, _ = load_data()

    X = train.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )
    y = train["label"].astype(int)

    # Group identical predictor vectors together.
    groups = pd.factorize(
        pd.MultiIndex.from_frame(X),
        sort=False
    )[0]

    print("\nSECTION 4: GROUP-AWARE CROSS-VALIDATION")
    print("=" * 55)
    print("Total observations:", len(X))
    print("Unique predictor groups:", len(np.unique(groups)))

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        print(f"\nTraining fold {fold} of 3...")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        overlap = len(
            set(groups[train_idx])
            & set(groups[val_idx])
        )

        assert overlap == 0, "Group leakage detected"
        assert y_train.nunique() == 2
        assert y_val.nunique() == 2

        model = Pipeline([
            (
                "preprocessor",
                clone(build_preprocessor(X_train))
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="log2",
                    random_state=42,
                    n_jobs=1
                )
            )
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        attack_idx = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_val)[
            :, attack_idx
        ]

        metrics = calculate_metrics(
            y_val, predictions, probabilities
        )

        row = {
            "Fold": fold,
            "Training Rows": len(X_train),
            "Validation Rows": len(X_val),
            "Training Attack Percentage": (
                100 * y_train.mean()
            ),
            "Validation Attack Percentage": (
                100 * y_val.mean()
            ),
            "Shared Predictor Groups": overlap,
            **metrics
        }

        results.append(row)

        print("Training rows:", len(X_train))
        print("Validation rows:", len(X_val))
        print("Shared predictor groups:", overlap)
        print("Validation attack percentage:",
              round(100 * y_val.mean(), 4))
        print("Accuracy:", round(metrics["Accuracy"], 4))
        print("Precision:", round(metrics["Precision"], 4))
        print("Recall:", round(metrics["Recall"], 4))
        print("F1 Score:", round(metrics["F1 Score"], 4))
        print("ROC-AUC:", round(metrics["ROC-AUC"], 4))
        print("False Positive Rate:",
              round(metrics["False Positive Rate"], 4))

    results_df = pd.DataFrame(results)

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Average Precision",
        "False Positive Rate",
        "False Negative Rate"
    ]

    summary = {}

    print("\nCROSS-VALIDATION SUMMARY")
    print("=" * 55)

    for metric in metric_columns:
        mean = results_df[metric].mean()
        std = results_df[metric].std(ddof=1)

        summary[metric] = {
            "mean": float(mean),
            "std": float(std)
        }

        print(f"{metric}: {mean:.4f} +/- {std:.4f}")

    results_df.to_csv(
        output_dir / "group_cv_fold_results.csv",
        index=False
    )

    with open(
        output_dir / "group_cv_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "method": "StratifiedGroupKFold",
                "folds": 3,
                "random_state": 42,
                "group_definition": "Identical predictors",
                "total_observations": int(len(X)),
                "unique_groups": int(
                    len(np.unique(groups))
                ),
                "summary": summary
            },
            file,
            indent=4
        )

    print("\nExperiment completed successfully!")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
