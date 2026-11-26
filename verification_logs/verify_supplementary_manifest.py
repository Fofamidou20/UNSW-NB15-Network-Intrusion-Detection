"""
Verify the supplementary SHA-256 evidence manifest
for the UNSW-NB15 computational reproducibility study.
"""

import csv
import hashlib
import json
import re
import sys

from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------
# 1. Repository and evidence paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "verification_logs"

MANIFEST = (
    EVIDENCE / "supplementary_evidence_manifest.csv"
)

RESULTS_FILE = (
    EVIDENCE /
    "supplementary_manifest_integrity_results.csv"
)

SUMMARY_FILE = (
    EVIDENCE /
    "supplementary_manifest_integrity_summary.json"
)

EXPECTED_FILE_COUNT = 31

EXPECTED_MANIFEST_SHA256 = (
    "6e22f1bcd5e47cb04cc74bbaf365e505e5d75"
    "ff7314299253db9cfb1a7fc95f6"
)


# --------------------------------------------------
# 2. SHA-256 calculation
# --------------------------------------------------

def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


# --------------------------------------------------
# 3. Validate supplementary manifest
# --------------------------------------------------

if not MANIFEST.is_file():

    print("ERROR: Supplementary manifest not found.")
    sys.exit(1)

actual_manifest_sha256 = calculate_sha256(MANIFEST)

manifest_hash_match = (
    actual_manifest_sha256
    == EXPECTED_MANIFEST_SHA256
)

print("=" * 75)
print("SUPPLEMENTARY SHA-256 INTEGRITY VERIFICATION")
print("=" * 75)

print(
    "Expected manifest SHA-256:",
    EXPECTED_MANIFEST_SHA256
)

print(
    "Actual manifest SHA-256:",
    actual_manifest_sha256
)

print("Manifest hash match:", manifest_hash_match)

if not manifest_hash_match:

    print("ERROR: Manifest fingerprint mismatch.")
    sys.exit(1)


# --------------------------------------------------
# 4. Read manifest records
# --------------------------------------------------

with MANIFEST.open(
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    expected_columns = [
        "filename",
        "size_bytes",
        "sha256",
    ]

    if reader.fieldnames != expected_columns:

        print("ERROR: Invalid manifest structure.")
        sys.exit(1)

    records = list(reader)

if len(records) != EXPECTED_FILE_COUNT:

    print("ERROR: Unexpected manifest file count.")
    print("Expected:", EXPECTED_FILE_COUNT)
    print("Observed:", len(records))
    sys.exit(1)

filenames = [
    record["filename"]
    for record in records
]

if len(filenames) != len(set(filenames)):

    print("ERROR: Duplicate manifest entries.")
    sys.exit(1)


# --------------------------------------------------
# 5. Verify individual evidence files
# --------------------------------------------------

results = []

for record in records:

    filename = record["filename"]

    relative_path = Path(filename)

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
    ):

        print("ERROR: Invalid evidence path:", filename)
        sys.exit(1)

    path = (ROOT / relative_path).resolve()

    if not path.is_relative_to(ROOT):

        print("ERROR: Evidence path outside repository.")
        sys.exit(1)

    try:

        expected_size = int(record["size_bytes"])

    except (ValueError, TypeError):

        print("ERROR: Invalid file size:", filename)
        sys.exit(1)

    expected_hash = record["sha256"].lower()

    if (
        expected_size < 0
        or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        )
    ):

        print("ERROR: Invalid manifest entry:", filename)
        sys.exit(1)

    file_exists = path.is_file()

    actual_size = None
    actual_hash = None

    if file_exists:

        actual_size = path.stat().st_size
        actual_hash = calculate_sha256(path)

    size_match = (
        file_exists
        and actual_size == expected_size
    )

    sha256_match = (
        file_exists
        and actual_hash == expected_hash
    )

    passed = (
        file_exists
        and size_match
        and sha256_match
    )

    status = "PASSED" if passed else "FAILED"

    results.append({

        "filename": filename,
        "file_exists": file_exists,
        "expected_size_bytes": expected_size,
        "actual_size_bytes": actual_size,
        "size_match": size_match,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "sha256_match": sha256_match,
        "status": status,

    })

    print(f"{status:6} | {filename}")


# --------------------------------------------------
# 6. Calculate verification statistics
# --------------------------------------------------

total = len(results)

passed_count = sum(
    record["status"] == "PASSED"
    for record in results
)

failed_count = total - passed_count

success_rate = (
    passed_count / total * 100
    if total else 0.0
)

overall_status = (

    "PASSED"

    if (
        total == EXPECTED_FILE_COUNT
        and failed_count == 0
        and manifest_hash_match
    )

    else "FAILED"
)


# --------------------------------------------------
# 7. Save detailed verification results
# --------------------------------------------------

with RESULTS_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# 8. Save consolidated verification summary
# --------------------------------------------------

summary = {

    "verification_timestamp_utc":
        datetime.now(timezone.utc).isoformat(),

    "manifest":
        MANIFEST.name,

    "expected_manifest_sha256":
        EXPECTED_MANIFEST_SHA256,

    "actual_manifest_sha256":
        actual_manifest_sha256,

    "manifest_hash_match":
        manifest_hash_match,

    "expected_files":
        EXPECTED_FILE_COUNT,

    "files_verified":
        total,

    "passed":
        passed_count,

    "failed":
        failed_count,

    "integrity_success_rate_percent":
        success_rate,

    "overall_status":
        overall_status,

    "verification_scope":
        "Current evidence files compared against "
        "the supplementary SHA-256 manifest",

}

with SUMMARY_FILE.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 9. Display final verification summary
# --------------------------------------------------

print("-" * 75)
print("SUPPLEMENTARY INTEGRITY VERIFICATION SUMMARY")
print("-" * 75)

print(json.dumps(summary, indent=4))

print("-" * 75)
print("Detailed results saved:", RESULTS_FILE)
print("Summary saved:", SUMMARY_FILE)

if overall_status == "PASSED":

    print("SUPPLEMENTARY MANIFEST VERIFICATION PASSED.")
    sys.exit(0)

else:

    print("SUPPLEMENTARY MANIFEST VERIFICATION FAILED.")
    sys.exit(1)