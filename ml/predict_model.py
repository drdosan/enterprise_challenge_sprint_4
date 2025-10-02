# -*- coding: utf-8 -*-
"""
Inferência do modelo de irrigação (pós-treino)

Modos de uso:
1) Padrão (lendo DB e pegando a última linha válida):
   python predict_model.py

   Parâmetros úteis:
   --minutes 30           # janela de leitura do DB (minutos)
   --model models/model.joblib

2) Manual (sem DB, passar todas as features):
   python predict_model.py --manual \
     --temperatura_solo 22.5 --qualidade_ar_ppm 410 \
     --temperatura_ar 26.1 --umidade_ar 55.3 \
     --umidade_solo 28.0 --luminosidade 300

Saída:
- Imprime no console um dicionário com:
  {
    "timestamp": "...",
    "features": {...},
    "pred": 0/1,
    "proba_irrigar": 0.XXX (se disponível),
    "regra_umidade_solo<limiar": 0/1
  }
- Salva também em reports/predict_last.json
"""

import os
import json
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pymysql

# ----------------- CONFIGS (mesmas do train) -----------------
DB_CFG = dict(
    host="192.185.217.50",
    user="qualidad_estufas",
    password="Padr@ao321",
    database="qualidad_estufas",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

SENSOR_CODES = {
    "DHT22_TEMP_AR_ESP32_01": "temp_ar",
    "DHT22_UMID_AR_ESP32_01": "umid_ar",
    "LM35_TEMP_SOLO_01": "temp_solo",
    "CAP_SOLO_ESP32_01": "umid_solo",
    "LDR_LUX_ESP32_01": "luz",
    "MQ135_CO2_ESP32_01": "co2",
}

DASH_TO_MODEL = {
    "temp_ar": "temperatura_ar",
    "umid_ar": "umidade_ar",
    "temp_solo": "temperatura_solo",
    "umid_solo": "umidade_solo",
    "luz": "luminosidade",
    "co2": "qualidade_ar_ppm",
}

FEATURES = [
    "temperatura_solo",
    "qualidade_ar_ppm",
    "temperatura_ar",
    "umidade_ar",
    "umidade_solo",
    "luminosidade",
]

TIME_BUCKET = "30s"         # igual ao train
UMID_SOLO_LIMIAR = 30.0     # mesma regra de referência (baseline)
MODEL_PATH = os.path.join("models", "model.joblib")
PREDICT_JSON = os.path.join("reports", "predict_last.json")


# ----------------- Utils -----------------
def _debug(msg: str):
    print(f"[DEBUG] {msg}")


def ensure_dirs():
    os.makedirs("reports", exist_ok=True)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    _debug(f"Carregando modelo de: {os.path.abspath(path)}")
    return joblib.load(path)


def load_from_db(minutes: int = 30) -> pd.DataFrame:
    sql = f"""
        SELECT
            COALESCE(s.codigo_sensor, CONCAT('sensor_', l.sensor_id)) AS codigo_sensor,
            l.valor,
            l.timestamp_leitura
        FROM leituras l
        LEFT JOIN sensores s ON s.id = l.sensor_id
        WHERE l.timestamp_leitura BETWEEN DATE_SUB(UTC_TIMESTAMP(), INTERVAL {int(minutes)} MINUTE)
                                      AND UTC_TIMESTAMP()
        ORDER BY l.timestamp_leitura ASC;
    """
    conn = pymysql.connect(**DB_CFG)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=["codigo_sensor", "valor", "timestamp_leitura"])
    # conversões robustas
    df["timestamp_leitura"] = pd.to_datetime(df["timestamp_leitura"], errors="coerce")
    df["valor"] = (
        df["valor"].astype(str).str.replace(",", ".", regex=False).str.strip()
    )
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["codigo_sensor"] = df["codigo_sensor"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp_leitura"])
    return df


def pivot_align(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot por sensor e alinha em grade temporal com forward-fill."""
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw[df_raw["codigo_sensor"].isin(SENSOR_CODES.keys())].copy()
    if df.empty:
        return df

    df["bucket"] = df["timestamp_leitura"].dt.floor(TIME_BUCKET)
    wide = df.pivot_table(
        index="bucket",
        columns="codigo_sensor",
        values="valor",
        aggfunc="last",
    ).sort_index()

    wide = wide.rename(columns=SENSOR_CODES)
    wide = wide.ffill(limit=10)
    wide = wide.reset_index().rename(columns={"bucket": "timestamp"})
    return wide


def build_features_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame apenas com as FEATURES nomeadas no padrão do modelo."""
    if wide.empty:
        return pd.DataFrame(columns=["timestamp"] + FEATURES)

    # cria colunas do modelo
    df = wide.copy()
    for dash_name, model_name in DASH_TO_MODEL.items():
        if dash_name in df.columns and model_name not in df.columns:
            df[model_name] = df[dash_name]
        elif model_name not in df.columns:
            df[model_name] = np.nan

    cols = ["timestamp"] + FEATURES
    df = df[[c for c in cols if c in df.columns]].copy()
    # remove linhas com features faltantes
    df = df.dropna(subset=FEATURES, how="any")
    return df


def get_last_sample(df_feat: pd.DataFrame):
    """Retorna (timestamp_str, features_dict, x_vector) da última linha válida."""
    if df_feat.empty:
        return None, None, None
    row = df_feat.iloc[-1]
    ts = (
        row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"])
    )
    feats = {f: float(row[f]) for f in FEATURES if f in df_feat.columns}
    x = np.array([feats[f] for f in FEATURES], dtype=float).reshape(1, -1)
    return ts, feats, x


def predict_dict(clf, feats: dict, x: np.ndarray):
    """Gera o payload de predição, incluindo probabilidade quando disponível."""
    pred = int(clf.predict(x)[0])
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = float(clf.predict_proba(x)[:, 1][0])

    regra = int(feats.get("umidade_solo", np.nan) < UMID_SOLO_LIMIAR) if "umidade_solo" in feats else None

    return pred, proba, regra


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Inferência do modelo de irrigação")
    ap.add_argument("--model", default=MODEL_PATH, help="Caminho do modelo .joblib")
    ap.add_argument("--minutes", type=int, default=30, help="Janela (minutos) para ler do DB")

    # modo manual
    ap.add_argument("--manual", action="store_true", help="Usar valores passados via CLI (sem DB)")
    for f in FEATURES:
        ap.add_argument(f"--{f}", type=float, help=f"Valor para {f} (modo manual)")

    return ap.parse_args()


def main():
    ensure_dirs()
    args = parse_args()
    clf = load_model(args.model)

    output = None

    if args.manual:
        # precisa de TODOS os valores das FEATURES
        missing = [f for f in FEATURES if getattr(args, f) is None]
        if missing:
            raise SystemExit(f"Faltam valores para (modo manual): {missing}")

        feats = {f: float(getattr(args, f)) for f in FEATURES}
        x = np.array([feats[f] for f in FEATURES], dtype=float).reshape(1, -1)
        pred, proba, regra = predict_dict(clf, feats, x)

        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "manual",
            "features": feats,
            "pred": pred,
            "proba_irrigar": proba,
            "regra_umidade_solo<limiar": regra,
        }

    else:
        # lê DB -> pivot -> features -> última linha válida
        _debug(f"Lendo DB (últimos {args.minutes} min)")
        df_raw = load_from_db(minutes=args.minutes)
        wide = pivot_align(df_raw)
        df_feat = build_features_frame(wide)
        ts, feats, x = get_last_sample(df_feat)

        if x is None:
            raise SystemExit("Sem amostra válida para previsão (faltam leituras/colunas).")

        pred, proba, regra = predict_dict(clf, feats, x)
        output = {
            "timestamp": ts,
            "source": "db_last_valid_row",
            "features": feats,
            "pred": pred,
            "proba_irrigar": proba,
            "regra_umidade_solo<limiar": regra,
        }

    # imprime no console
    print(json.dumps(output, ensure_ascii=False, indent=2))

    # salva em arquivo
    with open(PREDICT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    _debug(f"Predict salvo em: {os.path.abspath(PREDICT_JSON)}")


if __name__ == "__main__":
    main()
