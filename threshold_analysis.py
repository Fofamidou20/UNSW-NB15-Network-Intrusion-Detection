
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_loader import load_data


def run_threshold_analysis():

    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load training data only
    train, _ = load_data()

    X = train.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )
    y = train["label"].astype(int)

    # Stratified validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Load optimized model configuration
    saved_model = joblib.load(
        results_dir / "optimized_random_forest.joblib"
    )

    model = clone(saved_model)

    print("Training threshold validation model...")
    model.fit(X_train, y_train)

    # Predicted attack probabilities
    attack_index = list(
        model.classes_
    ).index(1)

    probabilities = model.predict_proba(
        X_val
    )[:, attack_index]

    # Threshold range
    thresholds = [round(x / 100, 2)
                  for x in range(10, 91, 5)]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(
                y_val, predictions
            ),
            "Precision": precision_score(
                y_val, predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_val, predictions,
                zero_division=0
            ),
            "False Positive Rate": (
                fp / (fp + tn)
                if fp + tn else 0
            ),
            "False Negatives": fn
        })

    threshold_df = pd.DataFrame(results)

    # Save complete threshold results
    threshold_df.to_csv(
        results_dir / "threshold_analysis.csv",
        index=False
    )

    # Identify highest F1-score
    best_index = threshold_df["F1 Score"].idxmax()
    best_result = threshold_df.loc[best_index]

    print("\nBest Validation Threshold:")
    print(best_result["Threshold"])

    print("\nPerformance at Best Threshold:")
    print(best_result.round(4).to_string())

    # Plot threshold performance
    plt.figure(figsize=(11, 7))

    for metric in [
        "Precision",
        "Recall",
        "F1 Score"
    ]:
        plt.plot(
            threshold_df["Threshold"],
            threshold_df[metric],
            marker="o",
            label=metric
        )

    plt.axvline(
        best_result["Threshold"],
        linestyle="--",
        color="gray",
        label="Best F1 Threshold"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Performance Score")
    plt.title(
        "Threshold Optimization - Random Forest"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        results_dir / "threshold_optimization.png",
        dpi=300
    )

    plt.close()

    print("\nThreshold analysis completed!")


if __name__ == "__main__":
    run_threshold_analysis()
