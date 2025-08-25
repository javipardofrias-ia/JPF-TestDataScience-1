# Capgemini/src/telco_churn/data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas: trim, espacios→'_', y evita barras."""
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("/", "_", regex=False)
    )
    return df


def load_telco_csv(
    path: str | Path,
    normalize_columns: bool = True,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Carga el CSV de Telco Churn y (opcionalmente) normaliza los nombres de columnas.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encuentra el fichero: {p}")

    try:
        df = pd.read_csv(p, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="latin-1")

    if normalize_columns:
        df = _normalize_columns(df)

    return df
