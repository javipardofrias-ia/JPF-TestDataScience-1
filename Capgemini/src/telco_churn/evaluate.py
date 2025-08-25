import pandas as pd
import numpy as np

def validate_telco_df(df: pd.DataFrame, strict: bool = False) -> dict:
    report = {"errors": [], "warnings": []}

    # ---------- columnas requeridas ----------
    required = {
        "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
        "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges",
        "TotalCharges","Churn",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        report["errors"].append(f"Faltan columnas requeridas: {missing}")

    # ---------- tipos esperados ----------
    num_should = ["tenure","MonthlyCharges","TotalCharges","Churn","SeniorCitizen"]
    for c in num_should:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            report["warnings"].append(f"{c}: dtype {df[c].dtype} (esperado numérico)")

    # ---------- nulos ----------
    nulls = df.isna().mean().sort_values(ascending=False)
    high = nulls[nulls > 0.05]
    if len(high):
        report["warnings"].append(f"Nulos >5% en: {list(high.index)}")
    if "Churn" in df.columns and df["Churn"].isna().any():
        report["errors"].append("Churn contiene nulos")

    # ---------- dominios básicos ----------
    def vals(col): return set(df[col].dropna().unique().tolist())

    # binarios Yes/No
    for c in ["Partner","Dependents","PhoneService","PaperlessBilling"]:
        if c in df.columns:
            bad = vals(c) - {"Yes","No"}
            if bad: report["warnings"].append(f"{c}: valores inesperados {bad}")

    # otras categóricas
    if "gender" in df.columns:
        bad = vals("gender") - {"Male","Female"}
        if bad: report["warnings"].append(f"gender: valores inesperados {bad}")

    if "SeniorCitizen" in df.columns:
        bad = set(pd.Series(df["SeniorCitizen"].dropna(), dtype=int).unique()) - {0,1}
        if bad: report["errors"].append(f"SeniorCitizen: valores no binarios {bad}")

    if "InternetService" in df.columns:
        bad = vals("InternetService") - {"DSL","Fiber optic","No"}
        if bad: report["warnings"].append(f"InternetService: {bad}")

    if "Contract" in df.columns:
        bad = vals("Contract") - {"Month-to-month","One year","Two year"}
        if bad: report["warnings"].append(f"Contract: {bad}")

    if "PaymentMethod" in df.columns:
        ok = {"Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"}
        bad = vals("PaymentMethod") - ok
        if bad: report["warnings"].append(f"PaymentMethod: {bad}")

    # ---------- rangos y coherencia ----------
    if "tenure" in df.columns and (df["tenure"] < 0).any():
        report["errors"].append("tenure tiene valores negativos")
    for c in ["MonthlyCharges","TotalCharges"]:
        if c in df.columns and (df[c] < 0).any():
            report["errors"].append(f"{c} tiene valores negativos")

    # relación aproximada TotalCharges ~ MonthlyCharges * tenure (solo aviso)
    if set(["TotalCharges","MonthlyCharges","tenure"]).issubset(df.columns):
        denom = (df["MonthlyCharges"] * df["tenure"]).replace(0, np.nan)
        ratio = (df["TotalCharges"] / denom).dropna()
        if ratio.quantile(0.99) > 2 or ratio.quantile(0.01) < 0.5:
            report["warnings"].append("Relación TotalCharges≈MonthlyCharges*tenure con outliers visibles")

    # ---------- cardinalidad/duplicados ----------
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    high_card = [c for c in cat_cols if df[c].nunique(dropna=True) > 100]
    if high_card:
        report["warnings"].append(f"Categóricas de alta cardinalidad (>100): {high_card}")

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        report["warnings"].append(f"Filas duplicadas: {dup_rows}")

    # ---------- modo estricto ----------
    if strict and report["errors"]:
        raise ValueError("Validación fallida: " + " | ".join(report["errors"]))
    return report

def pretty_print_report(report: dict) -> None:
    if report["errors"]:
        print("❌ ERRORES:")
        for e in report["errors"]:
            print("  -", e)
    if report["warnings"]:
        print("⚠️  WARNINGS:")
        for w in report["warnings"]:
            print("  -", w)
    if not report["errors"] and not report["warnings"]:
        print("✅ Sin incidencias")
