
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR / "data" /
    "UNSW_NB15_training-set.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" /
    "section4_eda"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load training dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("SECTION 4: NUMERICAL DISTRIBUTION ANALYSIS")
print("=" * 50)
print("Training dataset shape:", df.shape)

# --------------------------------------------------
# 3. Select numerical predictors
# --------------------------------------------------

excluded_columns = [
    "id",
    "label",
    "attack_cat"
]

numeric_features = [
    col for col in df.select_dtypes(
        include=np.number
    ).columns
    if col not in excluded_columns
]

print("Numerical predictors:", len(numeric_features))

# Selected network traffic features
selected_features = [
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "smean",
    "dmean"
]

selected_features = [
    col for col in selected_features
    if col in numeric_features
]

# --------------------------------------------------
# 4. Descriptive statistics
# --------------------------------------------------

statistics = df[numeric_features].describe().T

statistics["skewness"] = (
    df[numeric_features].skew()
)

statistics["missing_values"] = (
    df[numeric_features].isna().sum()
)

statistics.to_csv(
    OUTPUT_DIR /
    "numerical_descriptive_statistics.csv"
)

print("\nDescriptive statistics:")
print(statistics.loc[selected_features].round(4))

# --------------------------------------------------
# 5. Raw feature histograms
# --------------------------------------------------

ncols = 3
nrows = int(np.ceil(len(selected_features) / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(16, 4 * nrows)
)

axes = np.atleast_1d(axes).flatten()

for ax, feature in zip(axes, selected_features):

    values = df[feature].dropna()

    ax.hist(
        values,
        bins=50,
        color="steelblue",
        edgecolor="black",
        alpha=0.75
    )

    ax.set_title(feature)
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")

for ax in axes[len(selected_features):]:
    ax.set_visible(False)

fig.suptitle(
    "UNSW-NB15 Numerical Feature Distributions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_4_4_histograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 6. Log-transformed histograms
# --------------------------------------------------

nonnegative_features = [
    feature for feature in selected_features
    if (df[feature].dropna() >= 0).all()
]

nrows = int(np.ceil(len(nonnegative_features) / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(16, 4 * nrows)
)

axes = np.atleast_1d(axes).flatten()

for ax, feature in zip(axes, nonnegative_features):

    values = np.log1p(
        df[feature].dropna()
    )

    ax.hist(
        values,
        bins=50,
        color="darkorange",
        edgecolor="black",
        alpha=0.75
    )

    ax.set_title(feature + " (log1p)")
    ax.set_xlabel("log(1 + Feature Value)")
    ax.set_ylabel("Frequency")

for ax in axes[len(nonnegative_features):]:
    ax.set_visible(False)

fig.suptitle(
    "Log-Transformed Numerical Distributions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_4_5_log_histograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 7. Raw feature boxplots
# --------------------------------------------------

fig, axes = plt.subplots(
    len(selected_features),
    1,
    figsize=(12, 3 * len(selected_features))
)

axes = np.atleast_1d(axes)

for ax, feature in zip(axes, selected_features):

    ax.boxplot(
        df[feature].dropna(),
        vert=False,
        showfliers=True
    )

    ax.set_title(feature)
    ax.set_xlabel("Feature Value")

fig.suptitle(
    "UNSW-NB15 Numerical Feature Boxplots",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_4_6_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 8. Class-conditional distributions
# --------------------------------------------------

class_features = [
    "dur",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl"
]

class_features = [
    feature for feature in class_features
    if feature in numeric_features
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 9)
)

axes = axes.flatten()

for ax, feature in zip(axes, class_features):

    normal = df.loc[
        df["label"] == 0, feature
    ].dropna()

    attack = df.loc[
        df["label"] == 1, feature
    ].dropna()

    # Use shared bin edges for fair comparison
    combined = pd.concat([normal, attack])
    bins = np.histogram_bin_edges(
        combined, bins=50
    )

    ax.hist(
        normal,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Normal"
    )

    ax.hist(
        attack,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Attack"
    )

    ax.set_title(feature)
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Density")
    ax.legend()

for ax in axes[len(class_features):]:
    ax.set_visible(False)

fig.suptitle(
    "Normal vs Attack Feature Distributions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "figure_4_7_class_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 9. Outlier analysis using IQR
# --------------------------------------------------

outlier_records = []

for feature in numeric_features:

    values = df[feature].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (values < lower) |
        (values > upper)
    )

    outlier_records.append({
        "Feature": feature,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Outlier Count": int(outliers.sum()),
        "Outlier Percentage":
            100 * outliers.mean()
    })

outlier_df = pd.DataFrame(outlier_records)

outlier_df = outlier_df.sort_values(
    "Outlier Percentage",
    ascending=False
)

outlier_df.to_csv(
    OUTPUT_DIR / "numerical_outlier_summary.csv",
    index=False
)

print("\nTop 15 features by IQR outlier percentage:")
print(outlier_df.head(15).round(4).to_string(index=False))

# --------------------------------------------------
# 10. Save analysis metadata
# --------------------------------------------------

metadata = {
    "dataset": DATA_PATH.name,
    "rows": len(df),
    "columns": len(df.columns),
    "numeric_predictor_count": len(numeric_features),
    "selected_features": selected_features,
    "analysis_partition": "training_only",
    "log_transform_applied_to_model": False,
    "outliers_removed": False
}

with open(
    OUTPUT_DIR / "distribution_analysis_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nNumerical distribution analysis completed!")
print("Results saved to:", OUTPUT_DIR)
