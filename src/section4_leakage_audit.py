
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# SECTION 4: DUPLICATE AND DATA LEAKAGE AUDIT
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "section4_eda"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

train = pd.read_csv(
    DATA_DIR / "UNSW_NB15_training-set.csv"
)

test = pd.read_csv(
    DATA_DIR / "UNSW_NB15_testing-set.csv"
)

print("\nSECTION 4: DATA LEAKAGE AUDIT")
print("=" * 55)

# Remove the identifier, but retain the class label.
# attack_cat is retained here to detect annotation
# inconsistencies, not for model prediction.
columns = [
    c for c in train.columns if c != "id"
]

if set(train.columns) != set(test.columns):
    raise ValueError(
        "Training and testing columns differ."
    )

if "label" not in columns:
    raise ValueError("Binary label column missing.")

train_records = train[columns].copy()
test_records = test[columns].copy()

# --------------------------------------------------
# 1. DUPLICATES WITHIN EACH DATASET
# --------------------------------------------------

train_duplicates = int(
    train_records.duplicated().sum()
)

test_duplicates = int(
    test_records.duplicated().sum()
)

print("\nWITHIN-DATASET DUPLICATES")
print("Training duplicates:", train_duplicates)
print("Testing duplicates:", test_duplicates)

# --------------------------------------------------
# 2. TRAINING–TESTING OVERLAP
# --------------------------------------------------

unique_train = train_records.drop_duplicates()
unique_test = test_records.drop_duplicates()

overlap = unique_test.merge(
    unique_train,
    on=columns,
    how="inner"
)

overlap_count = len(overlap)

test_overlap_rows = int(
    test_records.merge(
        unique_train,
        on=columns,
        how="inner"
    ).shape[0]
)

print("\nTRAINING–TESTING OVERLAP")
print("Shared unique records:", overlap_count)
print("Testing rows matching training:",
      test_overlap_rows)

# --------------------------------------------------
# 3. RECREATE STRATIFIED VALIDATION SPLIT
# --------------------------------------------------

# This reproduces the documented 80/20 split
# if the original script used these exact settings.
model_train, validation = train_test_split(
    train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train["label"]
)

model_train_records = model_train[columns]
validation_records = validation[columns]

unique_model_train = (
    model_train_records.drop_duplicates()
)

validation_overlap = validation_records.merge(
    unique_model_train,
    on=columns,
    how="inner"
)

validation_overlap_count = len(validation_overlap)

print("\nTRAINING–VALIDATION OVERLAP")
print("Model training rows:", len(model_train))
print("Validation rows:", len(validation))
print(
    "Validation rows duplicated in training:",
    validation_overlap_count
)

validation_overlap_pct = (
    100 * validation_overlap_count / len(validation)
)

print(
    "Validation overlap percentage:",
    round(validation_overlap_pct, 4)
)

# --------------------------------------------------
# 4. DUPLICATE RECORDS WITH CONFLICTING LABELS
# --------------------------------------------------

# Check identical predictor combinations with
# different binary labels.
predictor_columns = [
    c for c in columns
    if c not in ("label", "attack_cat")
]

conflicting_groups = (
    train.groupby(
        predictor_columns,
        dropna=False
    )["label"]
    .nunique()
)

conflict_count = int(
    (conflicting_groups > 1).sum()
)

print("\nLABEL CONSISTENCY")
print(
    "Feature combinations with conflicting labels:",
    conflict_count
)

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

summary = {
    "training_rows": len(train),
    "testing_rows": len(test),
    "training_duplicates": train_duplicates,
    "testing_duplicates": test_duplicates,
    "training_testing_shared_unique_records":
        overlap_count,
    "testing_rows_matching_training":
        test_overlap_rows,
    "model_training_rows": len(model_train),
    "validation_rows": len(validation),
    "validation_rows_matching_training":
        validation_overlap_count,
    "validation_overlap_percentage":
        validation_overlap_pct,
    "conflicting_label_groups": conflict_count,
    "random_state": RANDOM_STATE
}

output_path = (
    RESULTS_DIR / "leakage_audit_summary.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

pd.DataFrame(
    [summary]
).to_csv(
    RESULTS_DIR / "leakage_audit_summary.csv",
    index=False
)

print("\nLeakage audit completed successfully!")
print("Results saved to:", RESULTS_DIR)
