
from pathlib import Path
import json
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Define original and reproduced result locations
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

TOLERANCE = 1e-10

# --------------------------------------------------
# 2. Load original and reproduced search summaries
# --------------------------------------------------

with open(
    ORIGINAL / "hyperparameter_search_summary.json",
    encoding="utf-8"
) as file:
    original_summary = json.load(file)

with open(
    REPRODUCED / "hyperparameter_search_summary.json",
    encoding="utf-8"
) as file:
    reproduced_summary = json.load(file)

parameters_match = (
    original_summary["best_parameters"]
    == reproduced_summary["best_parameters"]
)

original_f1 = original_summary["best_cv_f1"]
reproduced_f1 = reproduced_summary["best_cv_f1"]

best_f1_difference = abs(original_f1 - reproduced_f1)

configuration_match = all(
    original_summary[key] == reproduced_summary[key]
    for key in ["cv_folds", "n_candidates", "random_state"]
)

# --------------------------------------------------
# 3. Load candidate-level cross-validation results
# --------------------------------------------------

original = pd.read_csv(
    ORIGINAL / "hyperparameter_search_results.csv"
)

reproduced = pd.read_csv(
    REPRODUCED / "hyperparameter_search_results.csv"
)

parameter_columns = [
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__min_samples_split",
    "param_classifier__min_samples_leaf",
    "param_classifier__max_features"
]

metric_columns = [
    "split0_test_score",
    "split1_test_score",
    "split2_test_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
    "split0_train_score",
    "split1_train_score",
    "split2_train_score",
    "mean_train_score",
    "std_train_score"
]

original = original.sort_values(
    parameter_columns
).reset_index(drop=True)

reproduced = reproduced.sort_values(
    parameter_columns
).reset_index(drop=True)

candidate_count_match = (
    len(original) == len(reproduced) == 10
)

candidate_parameters_match = original[
    parameter_columns
].equals(reproduced[parameter_columns])

if not candidate_count_match or not candidate_parameters_match:
    raise ValueError(
        "Original and reproduced candidate configurations differ."
    )

# --------------------------------------------------
# 4. Compare all candidate-level numerical metrics
# --------------------------------------------------

records = []

for i in range(len(original)):

    for metric in metric_columns:

        original_value = float(original.loc[i, metric])
        reproduced_value = float(reproduced.loc[i, metric])

        difference = abs(
            original_value - reproduced_value
        )

        if metric == "rank_test_score":
            passed = original_value == reproduced_value
        else:
            passed = bool(
                np.isclose(
                    original_value,
                    reproduced_value,
                    atol=TOLERANCE,
                    rtol=0
                )
            )

        records.append({
            "candidate_parameters": str(
                original.loc[i, parameter_columns].to_dict()
            ),
            "metric": metric,
            "original_value": original_value,
            "reproduced_value": reproduced_value,
            "absolute_difference": difference,
            "status": "PASSED" if passed else "FAILED"
        })

comparison = pd.DataFrame(records)

# --------------------------------------------------
# 5. Calculate overall verification results
# --------------------------------------------------

total = len(comparison)

passed = int(
    (comparison["status"] == "PASSED").sum()
)

failed = total - passed

maximum_difference = float(
    comparison["absolute_difference"].max()
)

overall_passed = (
    parameters_match
    and configuration_match
    and candidate_count_match
    and candidate_parameters_match
    and best_f1_difference <= TOLERANCE
    and failed == 0
)

summary = {
    "best_parameters_match": parameters_match,
    "original_best_cv_f1": original_f1,
    "reproduced_best_cv_f1": reproduced_f1,
    "best_cv_f1_absolute_difference": best_f1_difference,
    "search_configuration_match": configuration_match,
    "candidate_count_match": candidate_count_match,
    "candidate_parameters_match": candidate_parameters_match,
    "candidates_compared": len(original),
    "metrics_per_candidate": len(metric_columns),
    "total_comparisons": total,
    "passed_comparisons": passed,
    "failed_comparisons": failed,
    "maximum_absolute_difference": maximum_difference,
    "numerical_tolerance": TOLERANCE,
    "timing_columns_excluded": True,
    "overall_status": (
        "PASSED" if overall_passed else "FAILED"
    )
}

# --------------------------------------------------
# 6. Save verification evidence
# --------------------------------------------------

comparison.to_csv(
    OUTPUT / "hyperparameter_reproduction_comparison.csv",
    index=False
)

with open(
    OUTPUT / "hyperparameter_reproduction_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

# --------------------------------------------------
# 7. Display verification results
# --------------------------------------------------

print("\nHYPERPARAMETER REPRODUCIBILITY VERIFICATION")

print(json.dumps(summary, indent=4))

print("\nFAILED COMPARISONS")

failures = comparison[
    comparison["status"] == "FAILED"
]

if failures.empty:
    print("None")
else:
    print(failures.to_string(index=False))

print("\nVerification evidence saved.")
