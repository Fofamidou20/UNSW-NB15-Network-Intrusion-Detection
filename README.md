# UNSW-NB15 Network Intrusion Detection

Hamidou Fofana | Monroe University | CS655 Machine Learning | November 2026

Comparative binary classification of normal (0) and attack (1) network traffic using Logistic Regression, Random Forest and a Multilayer Perceptron, including two optimized Random Forest configurations. The study evaluates exact predictor overlap, group-aware validation and generalization.

## Setup (Windows Anaconda Prompt)

Run commands from the repository root. The recorded full-run environment uses Python 3.11.18 and scikit-learn 1.9.1. The Windows environment export has had its machine-specific prefix removed. Package availability and clean installation have not been independently tested.

```bat
conda env create -f environment.yml
conda activate intrusion_repro
```

If the environment already exists, activate it rather than recreating it. Alternatively create a fresh environment with Python 3.11.18 and install `python -m pip install -r requirements.txt`. The requirements and lock files both contain the verified pinned pip dependencies. Do not silently substitute versions if exact reproduction is required.

## Dataset

See [data instructions](data/README.md). Download the official CSVs and verify their checksums before running:

```bat
python -c "from src.data_loader import load_data; load_data()"
```

Compare printed hashes with [dataset-checksums.txt](dataset-checksums.txt). The training and official test partitions contain 175,341 and 82,332 rows respectively. The pipeline excludes id, attack_cat and label from predictors, leaving 42 original predictors.

## Reproduce main results

The repository contains the original recorded outputs under `results/`. Run experiments in a separate extracted copy or clone: scripts overwrite results and model artifacts. Smoke tests are optional and must not replace final evidence. Execute each command only after the preceding command succeeds.

```bat
python -m pytest -q
python main.py
python -m src.hyperparameter_search
python -m src.final_evaluation
python -m src.reproducibility_check
python -m src.section4_eda
python -m src.section4_distributions
python -m src.section4_leakage_audit
python -m src.section4_overlap_reconciliation
python -m src.section4_leakage_controlled
python -m src.section4_group_cv
python -m src.section4_test_overlap
python -m src.section6_final_verification
```

`main.py` fits four pipelines and saves their joblib artifacts. Hyperparameter search clones the initially optimized RF pipeline and creates the CV-selected RF. Final evaluation and verification use these locally generated artifacts. The reproducibility check compares the saved final evaluation with predictions from the saved model, not an independent retraining. Compare newly generated outputs against the original committed results; differences may occur across environments. An optional `python main.py --smoke-test` runs a reduced dataset and overwrites main outputs; use only in a disposable copy before a full run.

Optional exploratory scripts (after `main.py`): `python -m src.feature_analysis`, `python -m src.feature_selection`, `python -m src.permutation_analysis`, `python -m src.threshold_analysis`. These are not required for the principal comparison; their exploratory outputs are not all included in the uploaded evidence.

## Recorded official-test results

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8097 | 0.8491 | 0.9556 |
| Random Forest | 0.8662 | 0.8905 | 0.9823 |
| MLP | 0.8591 | 0.8853 | 0.9803 |
| Initially optimized RF | 0.8677 | 0.8916 | 0.9814 |
| CV-selected RF | 0.8638 | 0.8887 | 0.9810 |

Source: [verified model metrics](results/section6_verification/verified_model_metrics.csv). Group-aware CV mean F1 = 0.9659; non-overlapping official-test F1 = 0.8760. The CV-selected RF has official-test FPR 0.2877. The original search uses ordinary stratified CV, not nested group-aware tuning. The official test was viewed during exploratory development, so it is not a pristine blind test. Exact predictor grouping does not ensure independent sessions or environments. Results are not evidence of deployment readiness.

## Repository contents

- `src/`: 23 original Python modules.
- `main.py`: main training entry point.
- `tests/`: original pipeline tests.
- `results/`: recorded experimental tables, metadata and 21 figures.
- `report/`: compiled research report and scholarly bibliography within it.
- `environment.yml`, `requirements.txt`, `requirements-lock.txt`: recorded environment specifications.

Raw datasets and serialized models are excluded. Only load joblib files you generated or trust. The dataset is credited to Moustafa and Slay (2015), UNSW-NB15, https://research.unsw.edu.au/projects/unsw-nb15-dataset. See the [research report](report/UNSW_NB15_Final_Research_Paper.pdf) for scholarly citations and methodological details.

## Packaging verification

The original two unit tests passed in the packaging environment (Python 3.13.5, scikit-learn 1.8.0). This is not the recorded experimental environment. Full retraining, exact dependency installation and external validation were not performed during packaging. No original Python algorithm code was changed.
