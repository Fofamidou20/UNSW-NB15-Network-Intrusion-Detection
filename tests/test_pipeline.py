import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor
from src.evaluate import metrics


def test_preprocessing_and_evaluation():
    x = pd.DataFrame({"number": [1, 2, 3, 4, 5, 6, 7, 8], "category": ["a", "b"] * 4})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    model = Pipeline([("preprocessing", build_preprocessor(x)), ("classifier", LogisticRegression())])
    model.fit(x, y)
    result = metrics(model, x, y)
    assert 0 <= result["f1"] <= 1
    assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == len(y)


def test_unknown_category():
    x = pd.DataFrame({"number": [1, 2], "category": ["a", "b"]})
    preprocessor = build_preprocessor(x)
    preprocessor.fit(x)
    preprocessor.transform(pd.DataFrame({"number": [3], "category": ["unseen"]}))
