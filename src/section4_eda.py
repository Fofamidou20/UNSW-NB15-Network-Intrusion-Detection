
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# SECTION 4: UNSW-NB15 EMPIRICAL DATA ANALYSIS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "section4_eda"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA / "UNSW_NB15_testing-set.csv"

RANDOM_STATE = 42

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("SECTION 4: EMPIRICAL DATA ANALYSIS")

    # 1. Load the official dataset partitions
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    assert "label" in train.columns
    assert "label" in test.columns
    assert set(train.columns) == set(test.columns)

    print("Training shape:", train.shape)
    print("Testing shape:", test.shape)

    # 2. Dataset structure
    structure = pd.DataFrame({
        "Feature": train.columns,
        "Data Type": train.dtypes.astype(str).values,
        "Unique Training Values": [
            train[c].nunique(dropna=True) for c in train.columns
        ]
    })
    structure.to_csv(OUT / "dataset_structure.csv", index=False)

    # 3. Class distribution
    records = []
    for name, df in [("Training", train), ("Testing", test)]:
        for label, count in df["label"].value_counts().sort_index().items():
            records.append({
                "Partition": name,
                "Label": int(label),
                "Class": "Attack" if label == 1 else "Normal",
                "Count": int(count),
                "Percentage": round(100 * count / len(df), 4)
            })

    class_df = pd.DataFrame(records)
    class_df.to_csv(OUT / "class_distribution.csv", index=False)
    print("\nCLASS DISTRIBUTION\n", class_df.to_string(index=False))

    pivot = class_df.pivot(
        index="Partition", columns="Class", values="Percentage"
    ).reindex(["Training", "Testing"])

    pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    plt.title("UNSW-NB15 Class Distribution")
    plt.ylabel("Percentage of Records")
    plt.xlabel("Dataset Partition")
    plt.ylim(0, 100)
    save_figure("figure_4_1_class_distribution.png")

    # 4. Missing and non-finite numerical values
    missing_records = []
    for name, df in [("Training", train), ("Testing", test)]:
        for col in df.columns:
            missing = int(df[col].isna().sum())
            infinite = 0
            if pd.api.types.is_numeric_dtype(df[col]):
                infinite = int(np.isinf(
                    df[col].to_numpy(dtype=float, na_value=np.nan)
                ).sum())
            missing_records.append({
                "Partition": name,
                "Feature": col,
                "Missing": missing,
                "Missing Percent": 100 * missing / len(df),
                "Infinite": infinite
            })

    missing_df = pd.DataFrame(missing_records)
    missing_df.to_csv(OUT / "missing_values.csv", index=False)

    missing_plot = missing_df.pivot(
        index="Feature", columns="Partition", values="Missing"
    )
    missing_plot = missing_plot.loc[
        missing_plot.sum(axis=1).sort_values(ascending=False).index
    ].head(15)

    missing_plot.plot(kind="barh", figsize=(10, 6))
    plt.title("Missing Values by Feature")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    save_figure("figure_4_2_missing_values.png")

    # 5. Descriptive statistics (training only)
    numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
    predictors = [
        c for c in numeric_cols
        if c not in ["id", "label"]
    ]

    desc = train[predictors].describe().T
    desc["median"] = train[predictors].median()
    desc.to_csv(OUT / "descriptive_statistics.csv")

    # 6. Histograms of selected numerical predictors
    selected = [
        c for c in ["dur", "sbytes", "dbytes", "rate", "sttl", "sload"]
        if c in predictors
    ]

    for col in selected:
        values = pd.to_numeric(train[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        # Use the 99th percentile as a display limit only.
        # No observations are removed from model training.
        upper = values.quantile(0.99)
        displayed = values[values <= upper]

        plt.figure(figsize=(8, 5))
        plt.hist(displayed, bins=50)
        plt.title(f"Distribution of {col} (up to 99th percentile)")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_figure(f"histogram_{col}.png")

    # 7. Boxplots comparing normal and attack traffic
    for col in selected:
        normal = pd.to_numeric(
            train.loc[train.label == 0, col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        attack = pd.to_numeric(
            train.loc[train.label == 1, col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        # Visualization-only clipping
        combined = pd.concat([normal, attack])
        lower, upper = combined.quantile([0.01, 0.99])

        plt.figure(figsize=(8, 5))
        plt.boxplot([
            normal.clip(lower, upper),
            attack.clip(lower, upper)
        ], tick_labels=["Normal", "Attack"])
        plt.title(f"{col}: Normal vs Attack (display clipped)")
        plt.ylabel(col)
        save_figure(f"boxplot_{col}.png")

    # 8. Correlation analysis
    corr = train[predictors].replace(
        [np.inf, -np.inf], np.nan
    ).corr()

    corr.to_csv(OUT / "correlation_matrix.csv")

    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr)), corr.index, fontsize=6)
    plt.title("UNSW-NB15 Numerical Feature Correlation")
    save_figure("figure_4_3_correlation_heatmap.png")

    # 9. Training versus testing feature comparison
    comparison = []
    for col in selected:
        for name, df in [("Training", train), ("Testing", test)]:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            comparison.append({
                "Feature": col,
                "Partition": name,
                "Mean": s.mean(),
                "Median": s.median(),
                "Std": s.std(),
                "P95": s.quantile(0.95)
            })

    pd.DataFrame(comparison).to_csv(
        OUT / "train_test_feature_comparison.csv", index=False
    )

    # 10. Verify the documented stratified split
    X = train.drop(
        columns=["id", "attack_cat", "label"], errors="ignore"
    )
    y = train["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20,
        stratify=y, random_state=RANDOM_STATE
    )

    split_rows = []
    for name, labels in [
        ("Model Training", y_train),
        ("Validation", y_val),
        ("Independent Test", test["label"])
    ]:
        split_rows.append({
            "Partition": name,
            "Rows": len(labels),
            "Normal": int((labels == 0).sum()),
            "Attack": int((labels == 1).sum()),
            "Attack Percent": round(100 * (labels == 1).mean(), 4)
        })

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT / "partition_summary.csv", index=False)
    print("\nPARTITION SUMMARY\n", split_df.to_string(index=False))

    # 11. Data quality summary
    quality = {
        "training_rows": len(train),
        "testing_rows": len(test),
        "training_columns": len(train.columns),
        "testing_columns": len(test.columns),
        "original_predictors": X.shape[1],
        "training_missing_values": int(train.isna().sum().sum()),
        "testing_missing_values": int(test.isna().sum().sum()),
        "training_duplicates_excluding_id": int(
            train.drop(columns=["id"], errors="ignore").duplicated().sum()
        ),
        "testing_duplicates_excluding_id": int(
            test.drop(columns=["id"], errors="ignore").duplicated().sum()
        ),
        "random_state": RANDOM_STATE
    }

    with open(OUT / "data_quality_summary.json", "w") as f:
        json.dump(quality, f, indent=4)

    print("\nDATA QUALITY\n", json.dumps(quality, indent=4))
    print("\nSection 4 analysis completed successfully!")
    print("Outputs saved to:", OUT)

if __name__ == "__main__":
    main()
