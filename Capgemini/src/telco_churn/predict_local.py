# src/telco_churn/predict_local.py
import json, requests

URL = "http://127.0.0.1:1234/invocations"
INPUT_JSON = r"C:\Users\Javichu\OneDrive - UFV\Escritorio\personal\Capgemini\Capgemini\src\telco_churn\input.json"

# 1) Lee tu input.json (que tiene "inputs": [ {...} ])
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2) Opción A — MLflow 2.x: dataframe_records
payload = {"dataframe_records": data.get("inputs", data.get("data", []))}
r = requests.post(URL, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
print("A) dataframe_records ->", r.status_code)
try:
    print("Pred:", r.json())
    raise SystemExit(0)
except Exception:
    pass  # si no es JSON o da error, probamos split

# 3) Opción B — dataframe_split (columnas explícitas)
columns = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
]
rows = [[rec[c] for c in columns] for rec in payload["dataframe_records"]]

payload_split = {"dataframe_split": {"columns": columns, "data": rows}}
r = requests.post(URL, json=payload_split, headers={"Content-Type":"application/json"}, timeout=30)
print("B) dataframe_split  ->", r.status_code)
try:
    print("Pred:", r.json())
except Exception:
    print("Body:", r.text[:500])

