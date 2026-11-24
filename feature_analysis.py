
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

def analyze_feature_importance():

    # Project directories
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"

    # Load optimized Random Forest
    model_path = results_dir / "optimized_random_forest.joblib"
    model = joblib.load(model_path)

    # Retrieve trained classifier and preprocessor
    rf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessing"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Extract importance scores
    importances = rf.feature_importances_

    # Create importance table
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save results
    importance_df.to_csv(
        results_dir / "feature_importance.csv",
        index=False
    )

    # Plot top 15 features
    top15 = importance_df.head(15).iloc[::-1]

    plt.figure(figsize=(11, 7))
    plt.barh(top15["Feature"], top15["Importance"])

    plt.xlabel("Feature Importance")
    plt.ylabel("Network Traffic Features")
    plt.title("Top 15 Features - Optimized Random Forest")
    plt.tight_layout()

    plt.savefig(
        results_dir / "feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\nTop 15 Important Features:")
    print(importance_df.head(15).to_string(index=False))

    print("\nFeature importance analysis completed!")


if __name__ == "__main__":
    analyze_feature_importance()
