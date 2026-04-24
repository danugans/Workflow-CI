import os
import joblib
import warnings
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Tracking URI (local)
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")

# Set experiment
mlflow.set_experiment("Eksperiment_Danu-setiawan")


# =========================
# LOAD DATA
# =========================
def load_data():
    X_train = joblib.load(os.path.join(DATA_DIR, "X_train.joblib"))
    X_test = joblib.load(os.path.join(DATA_DIR, "X_test.joblib"))
    y_train = joblib.load(os.path.join(DATA_DIR, "y_train.joblib"))
    y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


# =========================
# TRAIN MODEL
# =========================
def train():
    X_train, X_test, y_train, y_test = load_data()

    # ❗ TIDAK PERLU start_run() karena sudah dari MLflow Projects

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =========================
    # PREDIKSI
    # =========================
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    # =========================
    # METRICS
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))

    # =========================
    # LOG KE MLFLOW
    # =========================
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("threshold", threshold)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_train[:5]
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()