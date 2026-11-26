from pathlib import Path
import pandas as pd
import numpy as np
import json

# 1. Define verification directory

output_dir = Path("verification_logs")

# 2. Load original strict verification evidence

comparison = pd.read_csv(
    output_dir / "retrained_model_comparison.csv"
)

with open(
    output_dir / "retraining_verification_summary.json",
    "r"
) as file:
    strict_summary = json.load(file)

# 3. Define acceptance criteria

strict_tolerance = 1e-10
practical_tolerance = 1e-8

count_metrics = ["tn", "fp", "fn", "tp"]

assert len(comparison) == 44
assert not comparison.duplicated(
    ["model", "metric"]
).any()

# 4. Preserve original verification status

comparison = comparison.rename(
    columns={"status": "strict_status"}
)

# 5. Perform practical numerical comparison

is_count = comparison["metric"].isin(count_metrics)

practical_pass = np.where(
    is_count,
    comparison["original_value"]
    == comparison["reproduced_value"],
    np.isclose(
        comparison["original_value"],
        comparison["reproduced_value"],
        atol=practical_tolerance,
        rtol=0
    )
)

comparison["practical_status"] = np.where(
    practical_pass,
    "PASSED",
    "FAILED"
)

# 6. Calculate verification statistics

total = len(comparison)

strict_passed = int(
    (comparison["strict_status"] == "PASSED").sum()
)

practical_passed = int(
    (comparison["practical_status"] == "PASSED").sum()
)

practical_failed = total - practical_passed

counts_exact = bool(
    (
        comparison.loc[is_count, "original_value"]
        == comparison.loc[is_count, "reproduced_value"]
    ).all()
)

# 7. Generate verification summary

summary = {
    "models_compared": 4,
    "metrics_per_model": 11,
    "total_comparisons": total,
    "strict_tolerance": strict_tolerance,
    "strict_passed": strict_passed,
    "strict_failed": total - strict_passed,
    "strict_status": strict_summary["overall_status"],
    "practical_tolerance": practical_tolerance,
    "practical_passed": practical_passed,
    "practical_failed": practical_failed,
    "maximum_absolute_difference": float(
        comparison["absolute_difference"].max()
    ),
    "confusion_matrix_counts_exact": counts_exact,
    "confusion_matrix_totals_verified":
        strict_summary["confusion_matrix_totals_verified"],
    "byte_identical": strict_summary["byte_identical"],
    "practical_status": (
        "PASSED"
        if practical_failed == 0 and counts_exact
        else "FAILED"
    ),
    "tolerance_assessment": (
        "Post hoc numerical sensitivity analysis"
    ),
    "verification_scope": (
        "Four retrained models in a fresh repository "
        "clone using the existing experimental environment"
    )
}

# 8. Save separate verification evidence

comparison.to_csv(
    output_dir / "practical_reproducibility_comparison.csv",
    index=False
)

with open(
    output_dir / "practical_reproducibility_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)

# 9. Display strict numerical discrepancies

print("\nORIGINAL STRICT TOLERANCE DISCREPANCIES\n")

print(
    comparison.loc[
        comparison["strict_status"] == "FAILED",
        [
            "model",
            "metric",
            "absolute_difference",
            "strict_status",
            "practical_status"
        ]
    ].to_string(index=False)
)

# 10. Display final assessment

print("\nPRACTICAL REPRODUCIBILITY SUMMARY\n")
print(json.dumps(summary, indent=4))

if summary["practical_status"] != "PASSED":
    raise SystemExit(1)