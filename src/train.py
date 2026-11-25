import joblib
import pandas as pd
from .config import RESULTS
from .models import make_models
from .evaluate import metrics


def run_models(x_train, y_train, x_test, y_test):
    RESULTS.mkdir(exist_ok=True, parents=True)
    rows = []
    for name, model in make_models(x_train).items():
        print(f"Training {name}...", flush=True)
        model.fit(x_train, y_train)
        row = {"model": name, **metrics(model, x_test, y_test)}
        rows.append(row)
        joblib.dump(model, RESULTS / f"{name}.joblib")
        print(f"  test F1: {row['f1']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    return table
