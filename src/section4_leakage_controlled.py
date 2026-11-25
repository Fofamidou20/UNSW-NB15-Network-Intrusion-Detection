
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
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


def evaluate(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "Recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "F1 Score": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "ROC-AUC": float(roc_auc_score(
            y_true, probabilities
        )),
        "Average Precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "False Positive Rate": float(
            fp / (fp + tn) if fp + tn else 0
        ),
        "False Negative Rate": float(
            fn / (fn + tp) if fn + tp else 0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def run_experiment():

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = (
        project_dir / "results" / "section4_leakage_controlled"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train, _ = load_data()

    X = train.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )
    y = train["label"].astype(int)

    # Identical predictors must belong to the same group.
    # Grouping intentionally excludes all target information.
    groups = pd.factorize(
        pd.MultiIndex.from_frame(X),
        sort=False
    )[0]

    print("\nLEAKAGE-CONTROLLED VALIDATION")
    print("=" * 50)
    print("Total observations:", len(X))
    print("Unique predictor groups:", len(np.unique(groups)))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    overlap = len(train_groups.intersection(val_groups))

    print("\nPARTITION SUMMARY")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Training attack percentage:",
          round(100 * y_train.mean(), 4))
    print("Validation attack percentage:",
          round(100 * y_val.mean(), 4))
    print("Shared predictor groups:", overlap)

    assert overlap == 0, "Predictor group leakage detected!"
    assert y_train.nunique() == 2
    assert y_val.nunique() == 2

    # The preprocessing pipeline is fitted only on training data.
    preprocessor = build_preprocessor(X_train)

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        random_state=42,
        n_jobs=1
    )

    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    print("\nTraining leakage-controlled Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    attack_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_val)[:, attack_index]

    metrics = evaluate(
        y_val, predictions, probabilities
    )

    print("\nLEAKAGE-CONTROLLED RESULTS")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    summary = {
        "random_state": 42,
        "split_method": "GroupShuffleSplit",
        "group_definition": "Identical predictor values",
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "training_attack_percentage": float(
            100 * y_train.mean()
        ),
        "validation_attack_percentage": float(
            100 * y_val.mean()
        ),
        "shared_predictor_groups": overlap,
        "metrics": metrics
    }

    with open(
        output_dir / "leakage_controlled_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    pd.DataFrame([metrics]).to_csv(
        output_dir / "leakage_controlled_results.csv",
        index=False
    )

    print("\nExperiment completed successfully!")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    run_experiment()
