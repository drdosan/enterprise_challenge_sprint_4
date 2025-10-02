import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from joblib import dump
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ---------------------------
# Config
# ---------------------------
DB_HOST = "192.185.217.50"
DB_NAME = "qualidad_estufas"
DB_USER = "qualidad_estufas"
DB_PASS = "Padr@ao321"
DB_TABLE = "leituras"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === MAPEAMENTO SENSOR_ID -> FEATURE ===
# Ajuste conforme os seus sensores reais!
SENSOR_MAP = {
    101: "temperatura_ar",
    102: "umidade_ar",
    103: "temperatura_solo",
    104: "umidade_solo",
    105: "luminosidade",
    106: "qualidade_ar_ppm",
}


CANDIDATE_FEATURES = [
    "temperatura_ar",
    "umidade_ar",
    "temperatura_solo",
    "umidade_solo",
    "luminosidade",
    "qualidade_ar_ppm",
]

UMIDADE_SOLO_THRESHOLD = float("30")

# ---------------------------
# Carregar dados
# ---------------------------
print(f"Conectando ao MySQL em {DB_HOST}/{DB_NAME} (tabela {DB_TABLE}) ...")
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)

sql = f"""
SELECT id, sensor_id, valor, timestamp_leitura, raw_data, created_at
FROM `{DB_TABLE}`
WHERE timestamp_leitura IS NOT NULL
ORDER BY timestamp_leitura ASC
"""
df = pd.read_sql(sql, conn)
conn.close()

if df.empty:
    raise SystemExit("[ERRO] Nenhuma linha retornada de `leituras`.")

df["timestamp_leitura"] = pd.to_datetime(df["timestamp_leitura"], errors="coerce", utc=True)
df = df.dropna(subset=["timestamp_leitura"]).sort_values("timestamp_leitura")

# Mantém apenas sensores do mapa
df = df[df["sensor_id"].isin(SENSOR_MAP.keys())].copy()
if df.empty:
    raise SystemExit("[ERRO] Nenhuma leitura com sensor_id presente no SENSOR_MAP.")

# Pivot: index = timestamp, colunas = sensor_id, valores = valor
wide = (
    df.pivot_table(
        index="timestamp_leitura",
        columns="sensor_id",
        values="valor",
        aggfunc="last"
    )
    .sort_index()
)

# Renomeia colunas (sensor_id -> feature)
wide = wide.rename(columns=SENSOR_MAP)

# Se existirem colunas duplicadas, funde por média
if not wide.columns.is_unique:
    dup_names = wide.columns[wide.columns.duplicated()].unique().tolist()
    print(f"[AVISO] Colunas duplicadas detectadas e serão fundidas por média: {dup_names}")
    wide = wide.T.groupby(level=0).mean().T

# Reamostra para 1 minuto (opcional) + forward-fill curto
wide = wide.resample("1min").last().ffill(limit=5)

# Seleciona features existentes
features = [c for c in CANDIDATE_FEATURES if c in wide.columns]
if not features:
    raise SystemExit("[ERRO] Após o pivot, nenhuma feature conhecida ficou disponível. Revise SENSOR_MAP.")

# Label: pode_irrigar (solo seco)
if "umidade_solo" not in wide.columns:
    raise SystemExit("[ERRO] Não encontrei 'umidade_solo' após o pivot. Ajuste SENSOR_MAP para apontar o sensor correto.")

wide = wide.apply(pd.to_numeric, errors="coerce")
wide = wide.dropna(how="all")

y = (wide["umidade_solo"] < UMIDADE_SOLO_THRESHOLD).astype(int)
X = wide[features]

# ---------------------------
# Split treino/teste
# ---------------------------
cutoff_idx = int(len(wide) * 0.7)
X_train, X_test = X.iloc[:cutoff_idx], X.iloc[cutoff_idx:]
y_train, y_test = y.iloc[:cutoff_idx], y.iloc[cutoff_idx:]

# ---------------------------
# Pipeline de treino
# ---------------------------
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
])
preprocess = ColumnTransformer([("num", num_transformer, features)], remainder="drop")

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

pipe = Pipeline([("prep", preprocess), ("clf", clf)])

param_dist = {
    "clf__n_estimators": [200, 300, 500, 700],
    "clf__max_depth": [None, 5, 10, 15, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}

if len(X_train) >= 200:
    cv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_
else:
    best_model = pipe.fit(X_train, y_train)
    best_params = {}

# ---------------------------
# Avaliação
# ---------------------------
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred).tolist()

print("\n=== Classification Report ===\n", report)
print("Confusion Matrix [ [tn, fp], [fn, tp] ]:\n", cm)

# ---------------------------
# Salvar modelo e metadados
# ---------------------------
model_path = (MODEL_DIR / "model.joblib").resolve()
dump(best_model, model_path)

model_info = {
    "features": features,
    "umidade_solo_threshold": UMIDADE_SOLO_THRESHOLD,
    "best_params": best_params,
    "report": report,
    "confusion_matrix": cm,
    "versions": {
        "sklearn": __import__("sklearn").__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
}
info_path = (MODEL_DIR / "model_info.json").resolve()
with open(info_path, "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Modelo salvo em: {model_path}")
print(f"[OK] Metadados em: {info_path}")
