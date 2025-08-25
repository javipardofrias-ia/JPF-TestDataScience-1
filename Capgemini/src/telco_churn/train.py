import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from src.telco_churn.features import build_preprocessor

def train_and_log_model(
    df: pd.DataFrame,
    model,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    experiment_name="telco_churn_experiment"
):
    # Prepara datos
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Preprocesador
    preprocessor = build_preprocessor(X)

    # Pipeline
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # ------------------------------
    # MLflow (arreglo mínimo)
    # Genera una URI correcta tipo 'file:///C:/.../mlruns' usando as_uri()
    # ------------------------------
    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)  # opcional pero recomendado
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        # Entrenamiento
        clf.fit(X_train, y_train)

        # Predicciones
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Métricas
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log de parámetros y métricas
        mlflow.log_params(model.get_params())
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metrics({
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"]
        })

        # Log del modelo sin registrarlo
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(f"✅ Modelo '{model_name}' entrenado y loggeado con MLflow.")
