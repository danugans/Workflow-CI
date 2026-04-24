import os
import joblib
import warnings
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Setup MLflow tracking
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("Workflow_CI_Model_Danu_setiawan")


def load_dataset():
    X_train = joblib.load(os.path.join(DATA_DIR, "X_train.joblib"))
    X_test = joblib.load(os.path.join(DATA_DIR, "X_test.joblib"))
    y_train = joblib.load(os.path.join(DATA_DIR, "y_train.joblib"))
    y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


def run_training():
    X_train, X_test, y_train, y_test = load_dataset()

    # Autolog aktif (TANPA start_run)
    mlflow.sklearn.autolog()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc}")
    print(f"Precision: {prec}")
    print(f"Recall   : {rec}")
    print(f"F1 Score : {f1}")


if __name__ == "__main__":
    run_training()