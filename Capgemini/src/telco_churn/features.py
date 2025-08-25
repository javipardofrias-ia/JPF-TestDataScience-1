# Capgemini/src/telco_churn/features.py (añadir)
from __future__ import annotations
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza determinística usada en EDA y entrenamiento."""
    out = df.copy()
    # ID fuera
    out = out.drop(columns=["customerID"], errors="ignore")
    # TotalCharges a numérico
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    # target a 0/1 (si aún está en Yes/No)
    if out["Churn"].dtype == "O":
        out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    # unificar “No internet/phone service” -> “No”
    for c in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
        if c in out.columns:
            out[c] = out[c].replace({"No internet service": "No"})
    if "MultipleLines" in out.columns:
        out["MultipleLines"] = out["MultipleLines"].replace({"No phone service": "No"})
    # coerción a numérico
    for c in ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].fillna(0).astype(int)
    return out


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un preprocesador que escala numéricas, imputa y codifica categóricas.
    
    Args:
        df (pd.DataFrame): Dataset con las features (sin la columna objetivo)

    Returns:
        ColumnTransformer: pipeline de preprocesamiento
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Numéricas: imputar + escalar
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categóricas: imputar + one-hot
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor