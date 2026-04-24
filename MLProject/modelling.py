import os
import joblib
import warnings
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ✅ Gunakan BASE_DIR (WAJIB untuk CI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# ✅ Set tracking ke lokal folder project
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("Eksperiment_Danu_setiawan")


def load_data():
    try:
        X_train = joblib.load(os.path.join(DATA_DIR, "X_train.joblib"))
        X_test = joblib.load(os.path.join(DATA_DIR, "X_test.joblib"))
        y_train = joblib.load(os.path.join(DATA_DIR, "y_train.joblib"))
        y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    except Exception as e:
        raise FileNotFoundError(f"Dataset tidak ditemukan: {e}")
    
    return X_train, X_test, y_train, y_test


def train():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():

        # ✅ Model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)

        # ✅ Handle kemungkinan error predict_proba
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        # ✅ Evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1 Score :", round(f1, 4))

        # ✅ Logging ke MLflow
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # ✅ Log model + artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train[:5]
        )


if __name__ == "__main__":
    train()