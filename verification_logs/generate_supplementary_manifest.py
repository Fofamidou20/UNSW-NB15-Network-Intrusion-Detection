"""
Generate a supplementary SHA-256 manifest for the
UNSW-NB15 computational reproducibility evidence.

The original evidence manifest is preserved unchanged.
"""

import csv
import hashlib
import sys
from pathlib import Path


# --------------------------------------------------
# 1. Define repository and evidence paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "verification_logs"

MANIFEST = (
    EVIDENCE / "supplementary_evidence_manifest.csv"
)


# --------------------------------------------------
# 2. Exclude mutable and self-referential files
# --------------------------------------------------

EXCLUDED_FILES = {
    "supplementary_evidence_manifest.csv",
    "supplementary_manifest_integrity_results.csv",
    "supplementary_manifest_integrity_summary.json",
    "evidence_manifest_integrity_results.csv",
    "evidence_manifest_integrity_summary.json",
}

EXCLUDED_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
}


# --------------------------------------------------
# 3. Define required evidence files
# --------------------------------------------------

REQUIRED_FILES = [
    "original_baseline_commit.txt",
    "verification_evidence_manifest.csv",
    "consolidated_reproducibility_results.csv",
    "consolidated_reproducibility_summary.json",
    "consolidated_reproducibility_verification.py",
    "verify_evidence_manifest.py",
    "full_training_log.txt",
    "training_timing.txt",
    "hyperparameter_search_log.txt",
    "hyperparameter_search_timing.txt",
    "final_evaluation_log.txt",
    "final_evaluation_timing.txt",
    "reproduced_results/final_model_evaluation.csv",
    "reproduced_results/final_model_evaluation.json",
    "reproduced_results/hyperparameter_search_results.csv",
    "reproduced_results/model_comparison.csv",
]


# --------------------------------------------------
# 4. SHA-256 calculation
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
# 5. Validate required evidence
# --------------------------------------------------

missing_files = [
    name for name in REQUIRED_FILES
    if not (EVIDENCE / name).is_file()
]

if not (ROOT / ".gitattributes").is_file():
    missing_files.append(".gitattributes")

if missing_files:
    print("ERROR: Required evidence files are missing.")

    for name in missing_files:
        print(f"Missing: {name}")

    sys.exit(1)


# --------------------------------------------------
# 6. Collect evidence files
# --------------------------------------------------

evidence_files = []

for path in EVIDENCE.rglob("*"):

    if not path.is_file():
        continue

    if path.name in EXCLUDED_FILES:
        continue

    relative_path = path.relative_to(EVIDENCE)

    if any(
        part in EXCLUDED_DIRECTORIES
        for part in relative_path.parts
    ):
        continue

    evidence_files.append(path)

# Include the Git evidence-preservation configuration.
evidence_files.append(ROOT / ".gitattributes")

evidence_files.sort(
    key=lambda path: path.relative_to(ROOT).as_posix()
)


# --------------------------------------------------
# 7. Calculate evidence fingerprints
# --------------------------------------------------

records = []

for path in evidence_files:

    records.append({
        "filename": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    })


# --------------------------------------------------
# 8. Write supplementary manifest
# --------------------------------------------------

if MANIFEST.exists():
    print("ERROR: Supplementary manifest already exists.")
    print("Preserve the existing manifest before regeneration.")
    sys.exit(1)

with MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "size_bytes",
            "sha256",
        ],
    )

    writer.writeheader()
    writer.writerows(records)


# --------------------------------------------------
# 9. Display generation summary
# --------------------------------------------------

print("=" * 75)
print("SUPPLEMENTARY SHA-256 EVIDENCE MANIFEST")
print("=" * 75)

for record in records:
    print(
        f"{record['filename']} | "
        f"{record['size_bytes']} bytes"
    )

print("-" * 75)
print(f"Required evidence files: {len(REQUIRED_FILES)}")
print(f"Missing required files: {len(missing_files)}")
print(f"Total files fingerprinted: {len(records)}")
print(f"Manifest location: {MANIFEST}")
print(f"Manifest SHA-256: {calculate_sha256(MANIFEST)}")
print("Original evidence manifest preserved.")
print("SUPPLEMENTARY MANIFEST GENERATION COMPLETED.")