
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "verification_logs"

MANIFEST_PATH = LOG_DIR / "verification_evidence_manifest.csv"


# ============================================================
# 2. LOAD THE EXISTING SHA-256 MANIFEST
# ============================================================

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "filename",
    "size_bytes",
    "sha256"
}

if not required_columns.issubset(manifest.columns):
    raise ValueError("Invalid evidence manifest structure.")

if len(manifest) != 13:
    raise ValueError("Unexpected number of manifest records.")

if manifest["filename"].duplicated().any():
    raise ValueError("Duplicate filenames in manifest.")


# ============================================================
# 3. SHA-256 FUNCTION
# ============================================================

def calculate_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# 4. VERIFY EVERY EVIDENCE FILE
# ============================================================

records = []

for _, row in manifest.iterrows():

    filename = str(row["filename"])

    path = LOG_DIR / filename

    expected_size = int(row["size_bytes"])
    expected_hash = str(row["sha256"]).lower()

    exists = path.is_file()

    if exists:

        actual_size = path.stat().st_size
        actual_hash = calculate_sha256(path)

        size_match = actual_size == expected_size
        hash_match = actual_hash == expected_hash

    else:

        actual_size = None
        actual_hash = None

        size_match = False
        hash_match = False

    status = (
        "PASSED"
        if exists and size_match and hash_match
        else "FAILED"
    )

    records.append({
        "filename": filename,
        "file_exists": exists,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "size_match": size_match,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "sha256_match": hash_match,
        "status": status
    })


# ============================================================
# 5. CONSOLIDATE INTEGRITY RESULTS
# ============================================================

results = pd.DataFrame(records)

total = len(results)

passed = int(
    (results["status"] == "PASSED").sum()
)

failed = total - passed

summary = {
    "verification_timestamp_utc":
        datetime.now(timezone.utc).isoformat(),

    "manifest":
        MANIFEST_PATH.name,

    "files_verified": total,

    "passed": passed,

    "failed": failed,

    "integrity_success_rate_percent":
        passed / total * 100,

    "overall_status":
        "PASSED" if failed == 0 else "FAILED",

    "verification_scope":
        "Current evidence files compared against "
        "the previously generated SHA-256 manifest"
}


# ============================================================
# 6. SAVE VERIFICATION EVIDENCE
# ============================================================

results.to_csv(
    LOG_DIR / "evidence_manifest_integrity_results.csv",
    index=False
)

with open(
    LOG_DIR / "evidence_manifest_integrity_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\nSHA-256 EVIDENCE MANIFEST INTEGRITY VERIFICATION")

print(
    results[
        [
            "filename",
            "file_exists",
            "size_match",
            "sha256_match",
            "status"
        ]
    ].to_string(index=False)
)

print("\nINTEGRITY VERIFICATION SUMMARY")

print(json.dumps(summary, indent=4))

print("\nIntegrity verification evidence saved.")

if failed > 0:
    raise SystemExit(1)

print("\nEVIDENCE MANIFEST VERIFICATION PASSED.")
