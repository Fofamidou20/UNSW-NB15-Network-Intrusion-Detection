from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(frame):
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if len(numeric) + len(categorical) != frame.shape[1]:
        raise ValueError("Unexpected unsupported feature types")
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
