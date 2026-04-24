import os
import joblib
import warnings
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

DATA_DIR = "namadataset_preprocessing"

def load_data():
    X_train = joblib.load(os.path.join(DATA_DIR, "X_train.joblib"))
    X_test = joblib.load(os.path.join(DATA_DIR, "X_test.joblib"))
    y_train = joblib.load(os.path.join(DATA_DIR, "y_train.joblib"))
    y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


def train():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.3).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train()