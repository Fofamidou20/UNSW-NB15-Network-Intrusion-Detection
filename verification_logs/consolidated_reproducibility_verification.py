
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd


# ============================================================
# 1. PROJECT AND VERIFICATION DIRECTORIES
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "verification_logs"

STRICT_TOLERANCE = 1e-10
PRACTICAL_TOLERANCE = 1e-8


# ============================================================
# 2. REQUIRED EXPERIMENTAL EVIDENCE
# ============================================================

summary_files = {
    "retraining":
        "retraining_verification_summary.json",

    "practical":
        "practical_reproducibility_summary.json",

    "hyperparameter":
        "hyperparameter_reproduction_summary.json",

    "final":
        "final_evaluation_reproduction_summary.json"
}

comparison_files = [
    "retrained_model_comparison.csv",
    "practical_reproducibility_comparison.csv",
    "hyperparameter_reproduction_comparison.csv",
    "final_evaluation_reproduction_comparison.csv",
    "final_evaluation_file_hashes.csv"
]

verification_scripts = [
    "compare_retrained_models.py",
    "practical_reproducibility_check.py",
    "compare_hyperparameter_search.py",
    "compare_final_evaluation.py"
]

required_files = (
    list(summary_files.values())
    + comparison_files
    + verification_scripts
)


# ============================================================
# 3. CHECK EVIDENCE AVAILABILITY
# ============================================================

missing_files = [
    name for name in required_files
    if not (LOG_DIR / name).is_file()
]

if missing_files:
    raise FileNotFoundError(
        f"Missing verification evidence: {missing_files}"
    )

print("All required verification evidence files found.")


# ============================================================
# 4. LOAD ORIGINAL VERIFICATION SUMMARIES
# ============================================================

def load_json(filename):
    with open(
        LOG_DIR / filename,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


retraining = load_json(summary_files["retraining"])
practical = load_json(summary_files["practical"])
hyperparameter = load_json(summary_files["hyperparameter"])
final = load_json(summary_files["final"])


# ============================================================
# 5. VALIDATE CONSISTENCY BETWEEN VERIFICATION SUMMARIES
# ============================================================

def require(condition, message):
    if not condition:
        raise ValueError(message)


require(
    retraining["total_comparisons"]
    == practical["total_comparisons"],
    "Retraining comparison totals do not match."
)

require(
    retraining["passed_comparisons"]
    == practical["strict_passed"],
    "Strict retraining results do not match."
)

require(
    retraining["failed_comparisons"]
    == practical["strict_failed"],
    "Strict failure counts do not match."
)

require(
    retraining["numerical_tolerance"]
    == STRICT_TOLERANCE,
    "Unexpected retraining tolerance."
)

require(
    practical["strict_tolerance"]
    == STRICT_TOLERANCE,
    "Unexpected strict tolerance."
)

require(
    practical["practical_tolerance"]
    == PRACTICAL_TOLERANCE,
    "Unexpected practical tolerance."
)

require(
    hyperparameter["numerical_tolerance"]
    == STRICT_TOLERANCE,
    "Unexpected hyperparameter tolerance."
)

require(
    final["strict_tolerance"]
    == STRICT_TOLERANCE,
    "Unexpected final evaluation strict tolerance."
)

require(
    final["practical_tolerance"]
    == PRACTICAL_TOLERANCE,
    "Unexpected final evaluation practical tolerance."
)

require(
    hyperparameter["best_parameters_match"],
    "Best hyperparameters do not match."
)

require(
    hyperparameter["candidate_parameters_match"],
    "Hyperparameter candidates do not match."
)

require(
    hyperparameter["search_configuration_match"],
    "Search configurations do not match."
)

print("Verification summary consistency checks passed.")


# ============================================================
# 6. CONSOLIDATE EXPERIMENTAL COMPONENTS
# ============================================================

records = [
    {
        "component": "Four-model retraining",
        "total_comparisons":
            retraining["total_comparisons"],
        "strict_passed":
            retraining["passed_comparisons"],
        "strict_failed":
            retraining["failed_comparisons"],
        "practical_passed":
            practical["practical_passed"],
        "practical_failed":
            practical["practical_failed"],
        "maximum_absolute_difference":
            retraining["maximum_absolute_difference"]
    },
    {
        "component": "Hyperparameter search",
        "total_comparisons":
            hyperparameter["total_comparisons"],
        "strict_passed":
            hyperparameter["passed_comparisons"],
        "strict_failed":
            hyperparameter["failed_comparisons"],

        # All search comparisons were reproduced exactly.
        # Therefore, they also satisfy the looser tolerance.
        "practical_passed":
            hyperparameter["total_comparisons"],
        "practical_failed": 0,
        "maximum_absolute_difference":
            hyperparameter["maximum_absolute_difference"]
    },
    {
        "component": "Final CV-selected model",
        "total_comparisons":
            final["metrics_compared"],
        "strict_passed":
            final["strict_passed"],
        "strict_failed":
            final["strict_failed"],
        "practical_passed":
            final["practical_passed"],
        "practical_failed":
            final["practical_failed"],
        "maximum_absolute_difference":
            final["maximum_absolute_difference"]
    }
]

require(
    hyperparameter["maximum_absolute_difference"] == 0
    and hyperparameter["failed_comparisons"] == 0,
    "Hyperparameter results are not exactly reproduced."
)

table = pd.DataFrame(records)

for record in records:
    require(
        record["strict_passed"]
        + record["strict_failed"]
        == record["total_comparisons"],
        "Inconsistent strict comparison counts."
    )

    require(
        record["practical_passed"]
        + record["practical_failed"]
        == record["total_comparisons"],
        "Inconsistent practical comparison counts."
    )


# ============================================================
# 7. CALCULATE REPRODUCIBILITY RATES
# ============================================================

table["strict_success_rate_percent"] = (
    table["strict_passed"]
    / table["total_comparisons"]
    * 100
)

table["practical_success_rate_percent"] = (
    table["practical_passed"]
    / table["total_comparisons"]
    * 100
)

total_comparisons = int(
    table["total_comparisons"].sum()
)

strict_passed = int(table["strict_passed"].sum())
strict_failed = int(table["strict_failed"].sum())

practical_passed = int(table["practical_passed"].sum())
practical_failed = int(table["practical_failed"].sum())

strict_rate = strict_passed / total_comparisons * 100
practical_rate = practical_passed / total_comparisons * 100

maximum_difference = float(
    table["maximum_absolute_difference"].max()
)


# ============================================================
# 8. VERIFY CONFUSION MATRIX CONSISTENCY
# ============================================================

confusion_counts_exact = bool(
    practical["confusion_matrix_counts_exact"]
    and final["confusion_matrix_counts_exact"]
)

confusion_totals_verified = bool(
    retraining["confusion_matrix_totals_verified"]
    and final["confusion_matrix_totals_verified"]
)


# ============================================================
# 9. VERIFY FINAL OUTPUT FILE HASH RESULTS
# ============================================================

file_hashes = pd.read_csv(
    LOG_DIR / "final_evaluation_file_hashes.csv"
)

image_row = file_hashes.loc[
    file_hashes["file"] == "final_confusion_matrix.png"
]

require(
    len(image_row) == 1,
    "Final confusion matrix hash record is missing."
)

image_byte_identical = (
    str(image_row.iloc[0]["byte_identical"]).lower()
    == "true"
)


# ============================================================
# 10. CREATE CONSOLIDATED SUMMARY
# ============================================================

summary = {
    "verification_timestamp_utc":
        datetime.now(timezone.utc).isoformat(),

    "project":
        "UNSW-NB15 Machine Learning Intrusion Detection",

    "verification_scope":
        "Retraining and evaluation in a fresh repository "
        "clone using the existing experimental environment",

    "required_evidence_files": len(required_files),
    "missing_evidence_files": len(missing_files),

    "models_retrained":
        retraining["models_compared"] + 1,

    "experimental_components": len(records),

    "total_quantitative_comparisons":
        total_comparisons,

    "strict_numerical_tolerance":
        STRICT_TOLERANCE,

    "strict_passed": strict_passed,
    "strict_failed": strict_failed,

    "strict_success_rate_percent":
        strict_rate,

    "strict_status":
        "PASSED" if strict_failed == 0 else "FAILED",

    "practical_numerical_tolerance":
        PRACTICAL_TOLERANCE,

    "practical_passed": practical_passed,
    "practical_failed": practical_failed,

    "practical_success_rate_percent":
        practical_rate,

    "practical_status":
        "PASSED" if practical_failed == 0 else "FAILED",

    "maximum_absolute_difference":
        maximum_difference,

    "best_hyperparameters_match":
        hyperparameter["best_parameters_match"],

    "original_best_cv_f1":
        hyperparameter["original_best_cv_f1"],

    "reproduced_best_cv_f1":
        hyperparameter["reproduced_best_cv_f1"],

    "best_cv_f1_absolute_difference":
        hyperparameter["best_cv_f1_absolute_difference"],

    "hyperparameter_comparisons_exact":
        hyperparameter["maximum_absolute_difference"] == 0,

    "confusion_matrix_counts_exact":
        confusion_counts_exact,

    "confusion_matrix_totals_verified":
        confusion_totals_verified,

    "final_test_class_totals_verified":
        final["class_totals_verified"],

    "final_confusion_matrix_byte_identical":
        image_byte_identical,

    "all_final_output_files_byte_identical":
        final["all_output_files_byte_identical"],

    "complete_byte_level_reproducibility":
        False if (
            not retraining["byte_identical"]
            or not final["all_output_files_byte_identical"]
        ) else None,

    "practical_tolerance_assessment":
        "Post hoc numerical sensitivity analysis",

    "independent_software_environment_tested":
        False,

    "independent_researcher_replication_tested":
        False,

    "external_dataset_generalization_established":
        False,

    "production_deployment_readiness_established":
        False
}


# ============================================================
# 11. DETERMINE CONSOLIDATED ASSESSMENT
# ============================================================

if (
    practical_failed == 0
    and maximum_difference <= PRACTICAL_TOLERANCE
    and confusion_counts_exact
    and confusion_totals_verified
    and hyperparameter["overall_status"] == "PASSED"
    and final["strict_status"] == "PASSED"
):
    if strict_failed == 0:
        assessment = "STRICT_AND_PRACTICAL_REPRODUCIBILITY_PASSED"
    else:
        assessment = (
            "PRACTICAL_REPRODUCIBILITY_PASSED_"
            "STRICT_DISCREPANCIES_RETAINED"
        )
else:
    assessment = "FURTHER_INVESTIGATION_REQUIRED"

summary["consolidated_assessment"] = assessment


# ============================================================
# 12. GENERATE SHA-256 EVIDENCE MANIFEST
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


evidence_records = []

for filename in sorted(required_files):
    path = LOG_DIR / filename

    evidence_records.append({
        "filename": filename,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path)
    })

manifest = pd.DataFrame(evidence_records)


# ============================================================
# 13. SAVE CONSOLIDATED VERIFICATION EVIDENCE
# ============================================================

table.to_csv(
    LOG_DIR / "consolidated_reproducibility_results.csv",
    index=False
)

with open(
    LOG_DIR / "consolidated_reproducibility_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

manifest.to_csv(
    LOG_DIR / "verification_evidence_manifest.csv",
    index=False
)


# ============================================================
# 14. DISPLAY CONSOLIDATED RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CONSOLIDATED COMPUTATIONAL REPRODUCIBILITY VERIFICATION")
print("=" * 70)

print(table.to_string(index=False))

print("\nCONSOLIDATED VERIFICATION SUMMARY")
print(json.dumps(summary, indent=4))

print("\nGENERATED EVIDENCE FILES")

print("1. consolidated_reproducibility_results.csv")
print("2. consolidated_reproducibility_summary.json")
print("3. verification_evidence_manifest.csv")

print("\nCONSOLIDATED VERIFICATION COMPLETED.")
