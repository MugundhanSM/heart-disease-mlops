import os
from pathlib import Path
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import warnings
warnings.filterwarnings("ignore")

# ==================================================
# CI detection 
# ==================================================
IS_CI = os.getenv("CI", "false").lower() == "true"

# ==================================================
# Paths & MLflow configuration
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        PROJECT_ROOT / "data" / "processed" / "heart_disease_cleaned.csv",
    )
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:./mlruns",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Heart Disease Classification")

# ==================================================
# Load dataset
# ==================================================
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# ==================================================
# Feature groups
# ==================================================
categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]

# ==================================================
# Preprocessing + model pipeline
# ==================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# ==================================================
# Train-test split
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ==================================================
# Training
# ==================================================
print("Starting model training...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Training metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ==================================================
# MLflow logging (SKIPPED IN CI)
# ==================================================
if not IS_CI:
    with mlflow.start_run(run_name="Final Logistic Regression Model"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HeartDiseaseClassifier",
        )

    # ==================================================
    # Export model for containerized inference
    # ==================================================
    EXPORT_PATH = PROJECT_ROOT / "model_artifact"

    if EXPORT_PATH.exists():
        shutil.rmtree(EXPORT_PATH)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(EXPORT_PATH),
    )

    print(f"Model exported to {EXPORT_PATH}")

else:
    print("CI mode detected: Skipping MLflow logging and model export")

print("Training completed successfully.")