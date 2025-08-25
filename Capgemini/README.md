# Telco Customer Churn – Capgemini

Proyecto de **Machine Learning** para predecir la baja de clientes (“churn”) usando el dataset clásico *IBM Telco Customer Churn*. Incluye EDA, preparación de datos, entrenamiento con **MLflow**, y ejemplo de **serving** local del modelo.

> Generado automáticamente el 2025-08-25.

---

## 📦 Estructura del proyecto

```
Capgemini/
├─ .github/
  ├─ workflows/
    ├─ ci.yml
├─ .pre-commit-config.yaml
├─ README.md
├─ data/
  ├─ processed/
    ├─ telco_clean.csv
    ├─ telco_clean.parquet
    ├─ telco_sample_5k.csv
    ├─ telco_sample_5k.parquet
  ├─ raw/
    ├─ WA_Fn-UseC_-Telco-Customer-Churn.csv
├─ docs/
├─ mlruns/
  ├─ .trash/
  ├─ 242744980242275982/
    ├─ meta.yaml
├─ models/
├─ notebooks/
  ├─ 01_eda_modeling_telco.ipynb
  ├─ input.json
  ├─ mlruns/
            ├─ MLmodel
            ├─ requirements.txt
            ├─ MLmodel
            ├─ requirements.txt
            ├─ MLmodel
            ├─ requirements.txt
            ├─ MLmodel
            ├─ requirements.txt
            ├─ MLmodel
            ├─ requirements.txt
            ├─ MLmodel
            ├─ requirements.txt
            ├─ MLmodel
            ├─ requirements.txt
├─ requirements.txt
├─ scripts/
├─ serving/
  ├─ Dockerfile
  ├─ app.py
├─ src/
  ├─ telco_churn/
    ├─ __init__.py
    ├─ data.py
    ├─ eda.py
    ├─ evaluate.py
    ├─ features.py
    ├─ infer.py
    ├─ input.json
    ├─ predict_local.py
    ├─ train.py
    ├─ utils.py
    ├─ validation.py
├─ tests/
  ├─ test_smoke.py
```

---

## 🗂️ Datos

- **Raw:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`  
- **Procesado:** `data/processed/telco_clean.csv` (+ `.parquet`)  
- La columna objetivo es **`Churn`** (0/1).

**Columnas principales del dataset procesado (20):**  
`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn`

> Fuente del dataset: “IBM Telco Customer Churn”. Asegúrate de respetar su licencia de uso.

---

## 🧰 Requisitos

Python 3.11.9 recomendado. Instala dependencias:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (resumen):
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.0
lightgbm==4.5.0
xgboost==2.1.1
catboost==1.2.7
shap==0.45.1
mlflow==2.14.2
joblib==1.4.2
fastapi==0.111.0
uvicorn==0.30.5
pydantic==2.8.2
python-dotenv==1.0.1
```

---

## 📝 EDA y modelado

- Notebook principal: `notebooks/01_eda_modeling_telco.ipynb`  
- Código (WIP): `src/telco_churn/` contiene utilidades para carga, features y entrenamiento.

### Entrenamiento con MLflow (sus runs están versionados)

1) Arranca la UI de MLflow:
```bash
mlflow ui --backend-store-uri notebooks/mlruns
```
2) Ejecuta tu pipeline/notebook para registrar runs.  
3) Explora métricas y artefactos desde la UI.

**Mejor run detectado** (por AUC): `0fa067c4946048c4930fceea0d5f5dfc`
- ROC AUC: **0.828**
- F1-score: **0.779**
- Precision: **0.777**
- Recall: **0.789**

> Carpeta de experimentos: `notebooks/mlruns/666526178765312382/`

---

## 🚀 Serving (local) con MLflow

Puedes servir cualquier run como REST con `mlflow models serve`. Por ejemplo:

```bash
mlflow models serve \
  -m notebooks/mlruns/666526178765312382/0fa067c4946048c4930fceea0d5f5dfc/artifacts/model \
  -p 1234 --env-manager local
```

### Ejemplos de petición

**A) `dataframe_records`** (recomendado):
```bash
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d @src/telco_churn/input.json
```

**B) `dataframe_split`** (alternativo):
```bash
python src/telco_churn/predict_local.py
```

> El archivo `src/telco_churn/input.json` incluye un ejemplo con las columnas esperadas. Algunas features adicionales (`ExpectedTotal`, `Ratio`) son opcionales para pruebas.

---

## 🧪 Tests

Hay una prueba de humo tentativa en `tests/` *(pendiente de completar)*.  
Para ejecutarlas cuando existan:

```bash
pytest -q
```

---

## 🏗️ Estructura de código (WIP)

Módulos principales en `src/telco_churn/`:

- `data.py`: carga de datos y normalización de columnas.
- `features.py`: limpieza y `ColumnTransformer` para num/cat (imputación, escalado, one-hot).
- `train.py`: pipeline de entrenamiento + logging en MLflow.
- `evaluate.py`: validaciones básicas de schema/valores.
- `predict_local.py`: ejemplos de llamada a un endpoint de MLflow.
- `eda.py`, `utils.py`, `validation.py`: utilidades de análisis y validación (en construcción).

> La carpeta `serving/` contiene placeholders para un futuro **FastAPI** + **Dockerfile**.

---

## ▶️ Cómo reproducir (paso a paso)

```bash
# 1) Crear entorno e instalar deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) (Opcional) Explorar el EDA
jupyter notebook notebooks/01_eda_modeling_telco.ipynb

# 3) Lanzar MLflow UI para explorar runs
mlflow ui --backend-store-uri notebooks/mlruns

# 4) Servir el mejor modelo (ajusta EXP_ID / RUN_ID si fuera necesario)
mlflow models serve -m notebooks/mlruns/666526178765312382/0fa067c4946048c4930fceea0d5f5dfc/artifacts/model -p 1234 --env-manager local

# 5) Probar inferencia
curl -X POST http://127.0.0.1:1234/invocations -H "Content-Type: application/json" -d @src/telco_churn/input.json
```

---

## 📌 Roadmap / TODO

- Completar `serving/app.py` (FastAPI) y `serving/Dockerfile` para despliegue contenedorizado.
- Añadir tests unitarios y de integración reales (pytest).
- Incorporar validaciones de datos con `pydantic`/`pandera` o `Great Expectations`.
- Añadir selección de modelo (RF, XGBoost, LightGBM, CatBoost) y comparación en MLflow.
- Automatizar CI (`.github/workflows/ci.yml`) y `pre-commit`.
- Documentar métricas y explicabilidad (`shap`).

---

## 🪪 Licencia

- Código: MIT (o la que corresponda).  
- Datos: según licencia de *IBM Telco Customer Churn*.

---

## 🤝 Contribuir

1. Crea una rama desde `main`.
2. Añade cambios y tests.
3. Abre un PR describiendo tu aporte.

---

> ¿Quieres que deje listo el `FastAPI` + `Dockerfile` para servir el modelo? Puedo añadir una versión mínima funcional en `serving/` y los tests básicos.