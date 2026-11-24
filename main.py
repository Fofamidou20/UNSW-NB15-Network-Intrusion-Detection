import argparse
import json
import platform
import sys
import numpy
import pandas
import sklearn
from src.config import RESULTS
from src.data_loader import load_data, split_features
from src.train import run_models
from src.visualization import save_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Quick pipeline check using small stratified samples; not final results")
    args = parser.parse_args()
    train, test = load_data()
    x_train, y_train, x_test, y_test = split_features(train, test)
    if args.smoke_test:
        from sklearn.model_selection import train_test_split
        x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=2000, stratify=y_train, random_state=42)
        x_test, _, y_test, _ = train_test_split(x_test, y_test, train_size=500, stratify=y_test, random_state=42)
        print("SMOKE TEST ONLY — results are not comparable to full experiment")
    table = run_models(x_train, y_train, x_test, y_test)
    save_comparison(table, RESULTS / "model_comparison.png")
    versions = {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scikit_learn": sklearn.__version__, "smoke_test": args.smoke_test}
    (RESULTS / "run_metadata.json").write_text(json.dumps(versions, indent=2))
    print(table.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
