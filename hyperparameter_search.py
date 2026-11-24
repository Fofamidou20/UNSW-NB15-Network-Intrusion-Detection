
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

from src.data_loader import load_data


def run_hyperparameter_search():

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

    # Load existing pipeline configuration
    saved_pipeline = joblib.load(
        results_dir / "optimized_random_forest.joblib"
    )

    model = clone(saved_pipeline)

    # Hyperparameter search space
    parameter_grid = {
        "classifier__n_estimators": [80, 100, 121, 150],
        "classifier__max_depth": [15, 20, 25, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"]
    }

    # Reproducible cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_grid,
        n_iter=10,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=2,
        refit=True,
        return_train_score=True
    )

    print("Starting Random Forest hyperparameter search...")

    search.fit(X, y)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest Cross-Validation F1 Score:")
    print(round(search.best_score_, 6))

    # Save cross-validation results
    cv_results = pd.DataFrame(search.cv_results_)

    cv_results.to_csv(
        results_dir / "hyperparameter_search_results.csv",
        index=False
    )

    # Save best trained pipeline
    joblib.dump(
        search.best_estimator_,
        results_dir / "cv_optimized_random_forest.joblib"
    )

    # Save search summary
    summary = {
        "best_parameters": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "cv_folds": 3,
        "n_candidates": 10,
        "random_state": 42
    }

    with open(
        results_dir / "hyperparameter_search_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nHyperparameter search completed!")


if __name__ == "__main__":
    run_hyperparameter_search()
