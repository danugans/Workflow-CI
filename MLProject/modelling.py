import os
import joblib
import warnings
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "namadataset_preprocessing")
TRACKING_PATH = os.path.join(ROOT_DIR, "mlruns")

# Arahkan MLflow ke folder lokal
mlflow.set_tracking_uri(f"file:{TRACKING_PATH}")


def fetch_data():
    X_train = joblib.load(os.path.join(DATA_PATH, "X_train.joblib"))
    X_test = joblib.load(os.path.join(DATA_PATH, "X_test.joblib"))
    y_train = joblib.load(os.path.join(DATA_PATH, "y_train.joblib"))
    y_test = joblib.load(os.path.join(DATA_PATH, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


def build_and_evaluate():
    X_train, X_test, y_train, y_test = fetch_data()

    # Aktifkan autolog (tanpa start_run)
    mlflow.sklearn.autolog()

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs"
    )

    clf.fit(X_train, y_train)

    # Probabilitas + threshold custom (ini pembeda kamu)
    probs = clf.predict_proba(X_test)[:, 1]
    cut_off = 0.3
    preds = np.where(probs > cut_off, 1, 0)

    # Evaluasi manual (biar tetap ada kontrol)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Logging tambahan (autolog tidak cover threshold)
    mlflow.log_param("decision_threshold", cut_off)
    mlflow.log_metric("custom_f1", f1)


if __name__ == "__main__":
    build_and_evaluate()