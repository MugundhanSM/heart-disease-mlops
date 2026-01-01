from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# --------------------------------------------------
# Paths & MLflow setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_cleaned.csv"

mlflow.set_tracking_uri(f"file:///{PROJECT_ROOT}/mlruns")
mlflow.set_experiment("Heart Disease Classification")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# --------------------------------------------------
# Feature groups
# --------------------------------------------------
categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "thal"
]

numerical_features = [
    "age", "trestbps", "chol",
    "thalach", "oldpeak", "ca"
]

# --------------------------------------------------
# Preprocessing + model pipeline
# --------------------------------------------------
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

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Training & MLflow logging
# --------------------------------------------------
with mlflow.start_run(run_name="Final Logistic Regression Model"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="HeartDiseaseClassifier"
    )

print("Training completed successfully.")
