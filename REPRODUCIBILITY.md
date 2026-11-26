# Computational Reproducibility — UNSW-NB15 Network Intrusion Detection

## 1. Research project identification

This document records computational reproduction and evidence integrity verification for the UNSW-NB15 machine learning intrusion detection project, completed in November 2026.

## 2. Purpose and scope

Five models were retrained and evaluated in a fresh repository clone using the existing experimental software environment. The assessment covers four-model retraining, randomized hyperparameter search, and final evaluation of the cross-validation-selected Random Forest. It is not independent researcher replication or independent environment reconstruction.

## 3. Original repository baseline

Baseline commit: `c92b9e52fcd093c5b22aec58b2a47fe1448c979c`. The recorded baseline and HEAD matched before publication. The original tracked results were restored after reproduction, and reproduced tabular results were archived separately under `verification_logs/reproduced_results/`. Git status showed only new evidence and `.gitattributes` before documentation was added.

## 4. Computational environment

| Component | Observed version |
|---|---|
| OS | Microsoft Windows |
| Conda environment | intrusion_repro |
| Python | 3.11.18 |
| scikit-learn | 1.9.1 |
| NumPy | 2.4.6 |
| pandas | 3.1.1 |
| SciPy | 1.17.1 |
| joblib | 1.6.0 |
| Conda | 26.1.1 |
| Git | 2.56.1.windows.1 |

These versions were observed in the active verification environment. A separately reconstructed environment was not tested.

## 5. Dataset identification

The project uses the official UNSW-NB15 training and testing partitions, with binary attack detection as the prediction task. The final evaluation verified 82,332 test observations in both executions. Dataset fingerprints are not asserted by this verification summary; consult the project dataset audit for the original input checks.

## 6. Experimental reproduction methodology

The experimental workflow was re-executed in a fresh clone. Original and reproduced numerical outputs were compared using absolute differences. The original strict tolerance was 1e-10; a practical tolerance of 1e-8 was subsequently introduced as a post hoc sensitivity analysis. The strict result is retained unchanged. Timing columns were excluded from the hyperparameter numerical comparison.

## 7. Four-model retraining

Four models were compared across 11 metrics each, yielding 44 comparisons. Forty-two passed the strict criterion and two failed. All 44 passed the practical criterion. Maximum absolute difference: 2.8393463136211494e-09. Confusion matrix counts and totals matched. The comparison CSV files were not byte-identical.

Evidence: `verification_logs/retraining_verification_summary.json`, `verification_logs/retrained_model_comparison.csv`, and `verification_logs/practical_reproducibility_summary.json`.

## 8. Hyperparameter search reproduction

Ten candidates and eleven metrics per candidate yielded 110 comparisons, all with zero absolute difference. Search configuration, candidate count, candidate parameters, and best hyperparameters matched. The original and reproduced best cross-validation F1 scores were both 0.9693178753869902. Timing columns were excluded.

Evidence: `verification_logs/hyperparameter_reproduction_summary.json` and `verification_logs/hyperparameter_reproduction_comparison.csv`.

## 9. Final model evaluation

The retrained `cv_optimized_random_forest` passed all 12 comparisons under both tolerances. Maximum absolute difference: 2.3804069826383056e-11. Original and reproduced test totals were both 82,332. Confusion matrix counts, totals, and class totals matched. CSV and JSON were internally consistent within each execution according to the verification script. The confusion matrix PNG was byte-identical; the final CSV and JSON were not.

Evidence: `verification_logs/final_evaluation_reproduction_summary.json`, `verification_logs/final_evaluation_reproduction_comparison.csv`, and `verification_logs/final_evaluation_file_hashes.csv`.

## 10. Strict numerical reproducibility

| Component | Comparisons | Passed | Failed | Maximum absolute difference |
|---|---:|---:|---:|---:|
| Four-model retraining | 44 | 42 | 2 | 2.8393463136211494e-09 |
| Hyperparameter search | 110 | 110 | 0 | 0 |
| Final CV-selected model | 12 | 12 | 0 | 2.3804069826383056e-11 |
| Total | 166 | 164 | 2 | 2.8393463136211494e-09 |

Strict tolerance: 1e-10. Success rate: 98.79518072289156%. Overall strict status: **FAILED**. The two discrepancies are retained in the comparison evidence; their precise metric identities and causes are not asserted here.

## 11. Practical numerical reproducibility

Practical tolerance: 1e-8. All 166 comparisons passed (100%). This is a post hoc numerical sensitivity assessment, not a replacement for the original strict criterion. Small differences alone do not establish their computational cause.

## 12. Recorded execution times

| Stage | Duration |
|---|---:|
| Four-model training | 00:07:48.29 |
| Hyperparameter search | 08:47:18.26 |
| Final evaluation | 00:02:55.27 |
| Sum of logged intervals | 08:58:01.82 |

These intervals exclude gaps between stages and are not hardware-independent benchmarks. Timing logs are preserved under `verification_logs/`.

## 13. Original and reproduced results preservation

The original tracked results were restored from the baseline commit. The archived reproduced results include `final_model_evaluation.csv`, `final_model_evaluation.json`, `hyperparameter_search_results.csv`, and `model_comparison.csv`. Complete byte-level reproducibility was not achieved. Differences in file fingerprints are retained rather than normalized away.

## 14. Original SHA-256 evidence manifest

The original manifest, `verification_logs/verification_evidence_manifest.csv`, was preserved. Its verification reported 13 of 13 listed files passing existence, size, and SHA-256 checks, with exit code 0.

## 15. Supplementary SHA-256 evidence manifest

The supplementary manifest covers 31 files, including the original manifest, archived reproduced results, comparison scripts, summaries, and selected logs. All 16 required evidence files were present at generation. Verification on November 26, 2026 at 19:57:28 UTC reported 31 of 31 files passing, no failures, matching manifest hash, and exit code 0.

Manifest: `verification_logs/supplementary_evidence_manifest.csv`

SHA-256: `6e22f1bcd5e47cb04cc74bbaf365e505e5d75ff7314299253db9cfb1a7fc95f6`

Supplementary verifier SHA-256 (recorded separately): `37ba9d773722bf2142f9b418bbfdd8c1e9d4e72799e139e0d526cb58b2272b03`

The supplementary verifier and its generated reports were created after the supplementary manifest and are not covered by its 31-file result. The original and supplementary counts overlap and must not be added as unique files. Hash agreement establishes integrity relative to the recorded reference, not independent authentication.

## 16. Git evidence-preservation controls

The `.gitattributes` rule `verification_logs/** -text` disables Git text normalization for evidence files. The supplementary manifest also fingerprints `.gitattributes` itself. Do not regenerate manifests or modify archived evidence merely to obtain matching hashes. Publication and post-publication verification remain pending at the time of this document.

## 17. Verification instructions

From the repository root in Windows Anaconda Prompt, using the existing environment:

```bat
conda activate intrusion_repro
git status --short
certutil -hashfile verification_logs\supplementary_evidence_manifest.csv SHA256
certutil -hashfile verification_logs\verify_supplementary_manifest.py SHA256
python verification_logs\verify_evidence_manifest.py
echo Original verification exit code: %ERRORLEVEL%
python verification_logs\verify_supplementary_manifest.py
echo Supplementary verification exit code: %ERRORLEVEL%
```

Compare the fingerprints to Section 15 and expect exit code 0 for both scripts. Verification scripts may generate or refresh report files; run them in a disposable verification clone when preserving a published checkout. Review the project README and training scripts for the experimental execution workflow; exact retraining entry points are not reproduced here because they were not supplied in the current evidence. Do not run training over the preserved original results without first arranging separate output storage.

## 18. Methodological limitations

Two strict numerical discrepancies remain. The practical tolerance was selected post hoc. Complete byte-level equality was not achieved. Exact aggregate confusion matrix agreement does not prove identical predictions for every observation. No independent software environment, independent researcher replication, external dataset generalization, or production deployment readiness was established. Computational reproducibility does not resolve dataset overlap, potential leakage, or model selection limitations.

## 19. Consolidated assessment

| Dimension | Result |
|---|---|
| Strict numerical reproducibility | FAILED (164/166) |
| Practical numerical reproducibility | PASSED (166/166; post hoc) |
| Original evidence integrity | PASSED (13/13) |
| Supplementary evidence integrity | PASSED (31/31) |
| Complete byte-level reproducibility | Not established |
| Independent environment reconstruction | Not tested |
| Independent researcher replication | Not tested |
| External generalization | Not established |
| Production deployment readiness | Not established |

Consolidated evidence: `verification_logs/consolidated_reproducibility_summary.json` and `verification_logs/consolidated_reproducibility_results.csv`.

## 20. Final reproducibility statement

**PRACTICAL_REPRODUCIBILITY_PASSED_STRICT_DISCREPANCIES_RETAINED**

The experimental workflow demonstrated practical numerical reproducibility under the tested computational conditions. All hyperparameter comparisons and final model comparisons passed the original strict criterion, while two four-model retraining comparisons exceeded it. All 166 comparisons passed the subsequent practical sensitivity criterion. Both evidence manifests passed their respective integrity checks. These results support transparent and auditable computational reproduction within the stated scope, without implying universal determinism, independent replication, or operational validity.
