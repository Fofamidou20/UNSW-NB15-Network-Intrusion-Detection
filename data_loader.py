import hashlib
import pandas as pd
from .config import DATA


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data():
    paths = [DATA / "UNSW_NB15_training-set.csv", DATA / "UNSW_NB15_testing-set.csv"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset missing: {path}")
        print(f"SHA256 {path.name}: {checksum(path)}")
    train, test = [pd.read_csv(path) for path in paths]
    if "label" not in train or "label" not in test:
        raise ValueError("Both datasets must contain a label column")
    return train, test


def split_features(train, test):
    ignored = ["label", "attack_cat", "id"]
    x_train = train.drop(columns=ignored, errors="ignore")
    x_test = test.drop(columns=ignored, errors="ignore")
    if set(x_train.columns) != set(x_test.columns):
        raise ValueError("Training and test feature columns differ")
    if not set(train.label.unique()).issubset({0, 1}) or not set(test.label.unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1")
    return x_train, train.label.astype(int), x_test[x_train.columns], test.label.astype(int)
