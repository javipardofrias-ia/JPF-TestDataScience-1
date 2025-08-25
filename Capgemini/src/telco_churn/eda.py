import pandas as pd
import numpy as np

def num_summary_by_target(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """
    Devuelve un resumen estadístico (media, mediana, std, min, max) 
    de las variables numéricas agrupadas por la variable objetivo.

    Parámetros:
        df (pd.DataFrame): DataFrame con las variables.
        target (str): Nombre de la columna objetivo binaria (por defecto 'Churn').

    Retorna:
        pd.DataFrame: Resumen estadístico agrupado por target.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    
    grouped = df.groupby(target)[num_cols].agg(["mean", "median", "std", "min", "max"])
    return grouped.T  # Transpuesta para mejor visualización

import pandas as pd

def churn_rate_table(df: pd.DataFrame, col: str, top: int = 10) -> pd.DataFrame:
    """
    Calcula la tasa de churn y el número de observaciones por cada valor de una columna categórica.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        col (str): Nombre de la columna categórica a analizar.
        top (int): Número máximo de categorías a mostrar (ordenadas por tasa de churn).

    Retorna:
        pd.DataFrame: Tabla con 'rate' de churn y recuento 'n' por categoría.
    """
    if col not in df.columns:
        return pd.DataFrame()
    
    grouped = (
        df.groupby(col)["Churn"]
        .agg(rate="mean", n="count")
        .sort_values(["rate", "n"], ascending=[False, False])
    )
    
    return grouped.head(top)
