
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from src.data_loader import load_data
import joblib


def run_permutation_analysis():

    # Define project directories
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load original datasets
    train, _ = load_data()

    # Remove columns not used for prediction
    X = train.drop(
        columns=["id", "label", "attack_cat"],
        errors="ignore"
    )

    y = train["label"].astype(int)

    # Create stratified training-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Load optimized Random Forest pipeline
    saved_model = joblib.load(
        results_dir / "optimized_random_forest.joblib"
    )

    # Clone to obtain an unfitted model
    model = clone(saved_model)

    # Train using only the training subset
    print("Training validation model...")
    model.fit(X_train, y_train)

    # Select validation sample
    X_sample = X_val.sample(
        n=min(5000, len(X_val)),
        random_state=42
    )

    y_sample = y_val.loc[X_sample.index]

    # Calculate permutation importance
    print("Calculating permutation importance...")

    perm_result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=3,
        random_state=42,
        n_jobs=1
    )

    # Create results table
    perm_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Importance": perm_result.importances_mean,
        "Std": perm_result.importances_std
    })

    perm_df = perm_df.sort_values(
        "Importance",
        ascending=False
    )

    # Save results
    perm_df.to_csv(
        results_dir / "permutation_importance.csv",
        index=False
    )

    # Plot top 15 features
    top15 = perm_df.head(15).sort_values("Importance")

    plt.figure(figsize=(11, 7))

    plt.barh(
        top15["Feature"],
        top15["Importance"]
    )

    plt.xlabel("Mean Decrease in F1 Score")
    plt.ylabel("Network Traffic Features")
    plt.title("Top 15 Permutation Feature Importances")
    plt.tight_layout()

    plt.savefig(
        results_dir / "permutation_importance.png",
        dpi=300
    )

    plt.close()

    print("\nTop 15 Permutation Features:")
    print(perm_df.head(15).to_string(index=False))

    print("\nPermutation analysis completed!")


if __name__ == "__main__":
    run_permutation_analysis()
