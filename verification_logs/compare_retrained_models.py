from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
import json

# --------------------------------------------------
# 1. Define original and reproduced file locations
# --------------------------------------------------

original_path = (
    Path.home()
    / "Documents"
    / "UNSW_Original_Experimental_Evidence"
    / "results"
    / "model_comparison.csv"
)

reproduced_path = Path("results/model_comparison.csv")

output_dir = Path("verification_logs")
output_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load experimental results
# --------------------------------------------------

original = pd.read_csv(original_path)
reproduced = pd.read_csv(reproduced_path)

assert original["model"].is_unique
assert reproduced["model"].is_unique

assert set(original["model"]) == set(reproduced["model"])
assert set(original.columns) == set(reproduced.columns)

original = original.set_index("model").sort_index()
reproduced = reproduced.set_index("model").sort_index()

reproduced = reproduced[original.columns]

assert len(original) == 4

# --------------------------------------------------
# 3. Define evaluation metrics
# --------------------------------------------------

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_positive_rate"
]

count_metrics = ["tn", "fp", "fn", "tp"]

tolerance = 1e-10
comparison_results = []

# --------------------------------------------------
# 4. Compare every model and evaluation metric
# --------------------------------------------------

for model in original.index:

    for metric in metrics:

        old_value = float(original.loc[model, metric])
        new_value = float(reproduced.loc[model, metric])

        difference = abs(old_value - new_value)

        if metric in count_metrics:
            passed = old_value == new_value
        else:
            passed = np.isclose(
                old_value,
                new_value,
                atol=tolerance,
                rtol=0
            )

        comparison_results.append({
            "model": model,
            "metric": metric,
            "original_value": old_value,
            "reproduced_value": new_value,
            "absolute_difference": difference,
            "status": "PASSED" if passed else "FAILED"
        })

comparison = pd.DataFrame(comparison_results)

# --------------------------------------------------
# 5. Verify confusion matrix totals
# --------------------------------------------------

for frame in [original, reproduced]:

    assert (frame["tn"] + frame["fp"] == 37000).all()
    assert (frame["fn"] + frame["tp"] == 45332).all()

# --------------------------------------------------
# 6. Compare SHA-256 file hashes
# --------------------------------------------------

original_hash = hashlib.sha256(
    original_path.read_bytes()
).hexdigest()

reproduced_hash = hashlib.sha256(
    reproduced_path.read_bytes()
).hexdigest()

# --------------------------------------------------
# 7. Generate verification summary
# --------------------------------------------------

total = len(comparison)
passed = int((comparison["status"] == "PASSED").sum())
failed = total - passed

summary = {
    "models_compared": len(original),
    "metrics_per_model": len(metrics),
    "total_comparisons": total,
    "passed_comparisons": passed,
    "failed_comparisons": failed,
    "maximum_absolute_difference": float(
        comparison["absolute_difference"].max()
    ),
    "numerical_tolerance": tolerance,
    "original_sha256": original_hash,
    "reproduced_sha256": reproduced_hash,
    "byte_identical": original_hash == reproduced_hash,
    "confusion_matrix_totals_verified": True,
    "overall_status": "PASSED" if failed == 0 else "FAILED"
}

# --------------------------------------------------
# 8. Save verification evidence
# --------------------------------------------------

comparison.to_csv(
    output_dir / "retrained_model_comparison.csv",
    index=False
)

with open(
    output_dir / "retraining_verification_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)

# --------------------------------------------------
# 9. Display verification results
# --------------------------------------------------

print("\nQUANTITATIVE REPRODUCIBILITY VERIFICATION\n")

print(comparison.to_string(index=False))

print("\nVERIFICATION SUMMARY\n")

print(json.dumps(summary, indent=4))

if failed:
    raise SystemExit(1)