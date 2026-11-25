from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from .preprocessing import build_preprocessor
from .config import SEED


def make_models(x_train):
    constructors = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1),
        "neural_network": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=200, early_stopping=True, random_state=SEED),
        "optimized_random_forest": RandomForestClassifier(n_estimators=121, max_depth=25, min_samples_split=5, min_samples_leaf=1, max_features="sqrt", random_state=SEED, n_jobs=-1),
    }
    return {name: Pipeline([("preprocessing", build_preprocessor(x_train)), ("classifier", model)])
            for name, model in constructors.items()}
