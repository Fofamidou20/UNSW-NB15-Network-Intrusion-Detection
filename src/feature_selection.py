
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
    f1_score
)

from src.data_loader import load_data


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        )
    }


def run_feature_selection():

    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load training dataset only
    train, _ = load_data()

    X = train.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )
    y = train["label"].astype(int)

    # Independent validation partition
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Load saved model configuration
    saved_model = joblib.load(
        results_dir / "optimized_random_forest.joblib"
    )

    # Train full-feature model
    full_model = clone(saved_model)

    print("Training full-feature Random Forest...")
    full_model.fit(X_train, y_train)

    full_results = evaluate_model(
        full_model, X_val, y_val
    )

    # Select top 15 transformed features
    feature_names = (
        full_model.named_steps["preprocessing"]
        .get_feature_names_out()
    )

    importances = (
        full_model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    top_features = importance_df.head(15)["Feature"].tolist()

    # Identify original input columns
    original_columns = [
        name.split("__", 1)[1]
        for name in top_features
        if name.startswith("num__")
        and name.split("__", 1)[1] in X_train.columns
    ]

    original_columns = list(dict.fromkeys(original_columns))

    if not original_columns:
        raise ValueError(
            "No original numerical columns were selected."
        )

    print("\nSelected features:")
    print(original_columns)

    # Create reduced-feature pipeline
    reduced_model = clone(saved_model)

    # Replace preprocessing to use selected numeric columns
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    reduced_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                original_columns
            )
        ],
        remainder="drop"
    )

    reduced_model.set_params(
        preprocessing=reduced_preprocessor
    )

    print("\nTraining reduced-feature Random Forest...")

    reduced_model.fit(
        X_train[original_columns],
        y_train
    )

    reduced_results = evaluate_model(
        reduced_model,
        X_val[original_columns],
        y_val
    )

    # Compare performance
    comparison = pd.DataFrame([
        {"Model": "Full Features", **full_results},
        {"Model": "Top 15 Features", **reduced_results}
    ])

    comparison.to_csv(
        results_dir / "feature_selection_comparison.csv",
        index=False
    )

    print("\nFeature Selection Results:")
    print(comparison.round(4).to_string(index=False))

    # Save selected feature list
    pd.Series(
        original_columns,
        name="Selected Feature"
    ).to_csv(
        results_dir / "selected_features.csv",
        index=False
    )

    # Generate comparison chart
    ax = comparison.set_index("Model").plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        rot=0
    )

    ax.set_title("Full Features vs Top 15 Features")
    ax.set_ylabel("Performance Score")
    ax.set_xlabel("Random Forest Configuration")

    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        results_dir / "feature_selection_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nFeature selection experiment completed!")


if __name__ == "__main__":
    run_feature_selection()
