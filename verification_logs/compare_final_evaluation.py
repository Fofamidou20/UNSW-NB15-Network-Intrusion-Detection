
from pathlib import Path
import json
import hashlib
import math
import pandas as pd

# --------------------------------------------------
# 1. Directory configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL = (
    Path.home()
    / "Documents"
    / "UNSW_Original_Experimental_Evidence"
    / "results"
)

REPRODUCED = ROOT / "results"
OUTPUT = ROOT / "verification_logs"

OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Verification criteria
# --------------------------------------------------

STRICT_TOLERANCE = 1e-10
PRACTICAL_TOLERANCE = 1e-8
EXPECTED_TEST_SIZE = 82332

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "Average Precision",
    "False Positive Rate",
    "False Negative Rate",
    "True Negatives",
    "False Positives",
    "False Negatives",
    "True Positives"
]

COUNT_METRICS = [
    "True Negatives",
    "False Positives",
    "False Negatives",
    "True Positives"
]

# --------------------------------------------------
# 3. Utility functions
# --------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_metrics(data):
    if set(data.keys()) != set(METRICS):
        raise ValueError("Unexpected or missing metrics.")

    for metric in METRICS:
        value = float(data[metric])

        if not math.isfinite(value):
            raise ValueError(f"Non-finite value: {metric}")

        if metric in COUNT_METRICS:
            if value < 0 or not value.is_integer():
                raise ValueError(
                    f"Invalid confusion matrix count: {metric}"
                )
        elif not 0 <= value <= 1:
            raise ValueError(f"Invalid metric: {metric}")


# --------------------------------------------------
# 4. Load original and reproduced results
# --------------------------------------------------

original_json_path = (
    ORIGINAL / "final_model_evaluation.json"
)

reproduced_json_path = (
    REPRODUCED / "final_model_evaluation.json"
)

original = load_json(original_json_path)
reproduced = load_json(reproduced_json_path)

validate_metrics(original)
validate_metrics(reproduced)

# --------------------------------------------------
# 5. Compare all 12 evaluation measurements
# --------------------------------------------------

records = []

for metric in METRICS:

    original_value = float(original[metric])
    reproduced_value = float(reproduced[metric])

    difference = abs(
        original_value - reproduced_value
    )

    if metric in COUNT_METRICS:

        strict_pass = (
            original_value == reproduced_value
        )

        practical_pass = strict_pass

    else:

        strict_pass = (
            difference <= STRICT_TOLERANCE
        )

        practical_pass = (
            difference <= PRACTICAL_TOLERANCE
        )

    records.append({
        "metric": metric,
        "original_value": original_value,
        "reproduced_value": reproduced_value,
        "absolute_difference": difference,
        "strict_status": (
            "PASSED" if strict_pass else "FAILED"
        ),
        "practical_status": (
            "PASSED" if practical_pass else "FAILED"
        )
    })

comparison = pd.DataFrame(records)

# --------------------------------------------------
# 6. Verify confusion matrix counts and totals
# --------------------------------------------------

counts_exact = all(
    original[metric] == reproduced[metric]
    for metric in COUNT_METRICS
)

original_total = sum(
    int(original[metric])
    for metric in COUNT_METRICS
)

reproduced_total = sum(
    int(reproduced[metric])
    for metric in COUNT_METRICS
)

totals_verified = (
    original_total == EXPECTED_TEST_SIZE
    and reproduced_total == EXPECTED_TEST_SIZE
)

class_totals_verified = all(
    int(data["True Negatives"])
    + int(data["False Positives"]) == 37000
    and int(data["False Negatives"])
    + int(data["True Positives"]) == 45332
    for data in (original, reproduced)
)

# --------------------------------------------------
# 7. Verify CSV and JSON consistency
# --------------------------------------------------

def check_csv_json(directory, json_data):

    csv_data = pd.read_csv(
        directory / "final_model_evaluation.csv",
        float_precision="round_trip"
    )

    if len(csv_data) != 1:
        return False

    if set(csv_data.columns) != set(METRICS):
        return False

    for metric in METRICS:

        csv_value = float(csv_data.iloc[0][metric])
        json_value = float(json_data[metric])

        if metric in COUNT_METRICS:
            if csv_value != json_value:
                return False

        elif abs(csv_value - json_value) > 1e-15:
            return False

    return True


original_consistent = check_csv_json(
    ORIGINAL, original
)

reproduced_consistent = check_csv_json(
    REPRODUCED, reproduced
)

# --------------------------------------------------
# 8. File-level SHA-256 comparison
# --------------------------------------------------

file_records = []

for filename in [
    "final_model_evaluation.json",
    "final_model_evaluation.csv",
    "final_confusion_matrix.png"
]:

    original_hash = sha256_file(
        ORIGINAL / filename
    )

    reproduced_hash = sha256_file(
        REPRODUCED / filename
    )

    file_records.append({
        "file": filename,
        "original_sha256": original_hash,
        "reproduced_sha256": reproduced_hash,
        "byte_identical": (
            original_hash == reproduced_hash
        )
    })

file_comparison = pd.DataFrame(file_records)

# --------------------------------------------------
# 9. Construct verification summary
# --------------------------------------------------

strict_passed = int(
    (comparison["strict_status"] == "PASSED").sum()
)

practical_passed = int(
    (comparison["practical_status"] == "PASSED").sum()
)

total = len(comparison)

integrity_verified = (
    counts_exact
    and totals_verified
    and class_totals_verified
    and original_consistent
    and reproduced_consistent
)

summary = {
    "model": "cv_optimized_random_forest",
    "metrics_compared": total,

    "strict_tolerance": STRICT_TOLERANCE,
    "strict_passed": strict_passed,
    "strict_failed": total - strict_passed,

    "practical_tolerance": PRACTICAL_TOLERANCE,
    "practical_passed": practical_passed,
    "practical_failed": total - practical_passed,

    "maximum_absolute_difference": float(
        comparison["absolute_difference"].max()
    ),

    "confusion_matrix_counts_exact": counts_exact,
    "original_test_observations": original_total,
    "reproduced_test_observations": reproduced_total,
    "confusion_matrix_totals_verified": totals_verified,
    "class_totals_verified": class_totals_verified,

    "original_csv_json_consistent": original_consistent,
    "reproduced_csv_json_consistent": reproduced_consistent,

    "all_output_files_byte_identical": bool(
        file_comparison["byte_identical"].all()
    ),

    "strict_status": (
        "PASSED"
        if strict_passed == total and integrity_verified
        else "FAILED"
    ),

    "practical_status": (
        "PASSED"
        if practical_passed == total and integrity_verified
        else "FAILED"
    ),

    "tolerance_assessment": (
        "Original strict criterion and previously "
        "introduced practical sensitivity criterion"
    ),

    "verification_scope": (
        "Retrained cross-validation-selected Random "
        "Forest evaluated in a fresh repository clone "
        "using the existing experimental environment"
    )
}

# --------------------------------------------------
# 10. Save verification evidence
# --------------------------------------------------

comparison.to_csv(
    OUTPUT / "final_evaluation_reproduction_comparison.csv",
    index=False
)

file_comparison.to_csv(
    OUTPUT / "final_evaluation_file_hashes.csv",
    index=False
)

with open(
    OUTPUT / "final_evaluation_reproduction_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)

# --------------------------------------------------
# 11. Display verification results
# --------------------------------------------------

print("\nFINAL MODEL QUANTITATIVE VERIFICATION\n")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.15g}"
    )
)

print("\nSTRICT TOLERANCE DISCREPANCIES\n")

discrepancies = comparison[
    comparison["strict_status"] == "FAILED"
]

if discrepancies.empty:
    print("None")
else:
    print(discrepancies.to_string(index=False))

print("\nFILE-LEVEL REPRODUCIBILITY\n")

print(
    file_comparison[
        ["file", "byte_identical"]
    ].to_string(index=False)
)

print("\nFINAL REPRODUCIBILITY SUMMARY\n")

print(json.dumps(summary, indent=4))

print("\nVerification evidence saved.")

if summary["practical_status"] != "PASSED":
    raise SystemExit(1)
