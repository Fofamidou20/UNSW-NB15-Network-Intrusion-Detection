
import pandas as pd
from pathlib import Path

from src.data_loader import load_data

def run_reconciliation():
    train, test = load_data()

    # Complete records, excluding identifier
    full_columns = [
        c for c in train.columns
        if c != "id"
    ]

    # Model predictor columns
    predictor_columns = [
        c for c in train.columns
        if c not in ["id", "label", "attack_cat"]
    ]

    def membership_mask(columns):
        train_index = pd.MultiIndex.from_frame(
            train[columns].drop_duplicates()
        )
        test_index = pd.MultiIndex.from_frame(
            test[columns]
        )
        return test_index.isin(train_index)

    full_overlap = membership_mask(full_columns)
    predictor_overlap = membership_mask(
        predictor_columns
    )

    print("\nOVERLAP RECONCILIATION")
    print("=" * 50)

    print("Total testing rows:", len(test))

    print(
        "Complete-record overlap:",
        int(full_overlap.sum())
    )

    print(
        "Predictor overlap:",
        int(predictor_overlap.sum())
    )

    print(
        "Predictor-only additional overlap:",
        int((predictor_overlap & ~full_overlap).sum())
    )

    print(
        "Full overlap outside predictor overlap:",
        int((full_overlap & ~predictor_overlap).sum())
    )

    assert not (full_overlap & ~predictor_overlap).any(), (
        "Unexpected mismatch between overlap definitions"
    )

    output_dir = (
        Path("results") / "section4_overlap_reconciliation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "total_test_rows": len(test),
        "complete_record_overlap": int(full_overlap.sum()),
        "predictor_overlap": int(predictor_overlap.sum()),
        "additional_predictor_overlap": int(
            (predictor_overlap & ~full_overlap).sum()
        )
    }])

    summary.to_csv(
        output_dir / "overlap_reconciliation.csv",
        index=False
    )

    print("\nReconciliation completed successfully!")

if __name__ == "__main__":
    run_reconciliation()
