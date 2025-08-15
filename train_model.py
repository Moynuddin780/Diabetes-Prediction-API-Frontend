import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("E:\Jupyter Programs\diabetes.csv")  # place dataset here
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH.resolve()}. "
            "Please download Pima Indians Diabetes dataset as 'diabetes.csv'."
        )
    df = pd.read_csv(DATA_PATH)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    return df

def build_models() -> Dict[str, Any]:
    """
    Build multiple candidate models (at least two).
    We scale features where needed using a Pipeline.
    """
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11)),
        ]),
        "dt": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    }
    return models

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

def main():
    df = load_data()
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models()
    results = {}

    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"[{name}] -> {metrics}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name

    assert best_name is not None, "No best model selected."
    best_model = models[best_name]

    # Persist best model and artifacts
    model_path = MODELS_DIR / "diabetes_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model '{best_name}' to {model_path}")

    metrics_out = {
        "selected_model": best_name,
        "metrics_per_model": results,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(MODELS_DIR / "feature_order.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    print("Artifacts saved to 'models/' directory.")

if __name__ == "__main__":
    main()