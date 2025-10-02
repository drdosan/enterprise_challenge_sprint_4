from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
import pymysql, json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load

# ======== CONFIG BANCO ========
DB_CFG = dict(
    host='192.185.217.50',
    user='qualidad_estufas',
    password='Padr@ao321',
    database='qualidad_estufas',
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

# ======== ML – Caminhos ========
MODEL_DIR = Path("../machine_learning/models")
MODEL_PATH = MODEL_DIR / "model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"

# ======== MAPA DE SENSORES ========
# (ajuste se necessário)
SENSOR_MAP = {
    101: "temperatura_ar",
    102: "umidade_ar",
    103: "temperatura_solo",
    104: "umidade_solo",
    105: "luminosidade",
    106: "qualidade_ar_ppm",
}

app = Flask(__name__)
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "API Estufa Inteligente",
        "description": "Endpoints mínimos para ingestão de leituras do ESP32, status e predição ML.",
        "version": "1.1.0"
    },
    "schemes": ["http"],
    "basePath": "/"
})

# ========= Helpers =========
def db():
    return pymysql.connect(**DB_CFG)

def sensor_id_por_codigo(codigo: str) -> int:
    """Resolve o ID do sensor pela coluna única `codigo_sensor`."""
    codigo = (codigo or "").strip()
    if not codigo:
        raise ValueError("codigo_sensor vazio")
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sensores WHERE codigo_sensor=%s", (codigo,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Sensor não cadastrado: {codigo}")
            return int(row["id"])

# ML lazy-load
_model = None
_features = None
_model_info = None

def load_model_if_needed():
    global _model, _features, _model_info
    if _model is None or _features is None:
        if not MODEL_PATH.exists() or not INFO_PATH.exists():
            raise RuntimeError("Modelo não encontrado. Treine antes de usar a predição.")
        _model = load(MODEL_PATH)
        _model_info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
        _features = _model_info.get("features", [])
        if not _features:
            raise RuntimeError("Arquivo model_info.json não contém a lista de 'features'.")
    return _model, _features, _model_info

def predict_df(df: pd.DataFrame):
    """Executa a predição no DataFrame; retorna lista de dicts."""
    model, features, _ = load_model_if_needed()
    X = df.reindex(columns=features)  # faltantes -> NaN (pipeline deve imputar)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = None
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            proba = 1 / (1 + np.exp(-scores))
        except Exception:
            proba = None
    pred = model.predict(X)
    out = []
    for i in range(len(X)):
        item = {"index": int(i), "classe": int(pred[i])}
        if proba is not None:
            item["probabilidade_irrigar"] = float(proba[i])
        out.append(item)
    return out

# ========= Endpoints =========
@app.get("/health")
@swag_from({"tags": ["Infra"], "responses": {200: {"description": "OK"}}})
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}, 200

@app.get("/status-irrigacao")
@swag_from({
    "tags": ["Irrigacao"],
    "responses": {200: {"description": "Indica se pode irrigar agora", "schema": {
        "type": "object", "properties": {"pode_irrigar": {"type": "boolean"}, "fonte": {"type": "string"}}
    }}}
})
def status_irrigacao():
    # Fallback simples (sem serviços externos): irrigar se não chover (desconhecido) → True
    return jsonify({"pode_irrigar": True, "fonte": "fallback"}), 200

@app.post("/leituras/batch")
@swag_from({
    "tags": ["Leituras"],
    "parameters": [{
        "name": "body", "in": "body", "required": True,
        "schema": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "codigo_sensor": {"type": "string"},
                "valor": {"type": "number"},
                "ts": {"type": "string", "description": "ISO8601 opcional; se ausente, servidor usa NOW()"},
                "raw_data": {"type": "object"}
            },
            "required": ["codigo_sensor", "valor"]
        }}
    }],
    "responses": {
        201: {"description": "Inserções realizadas", "schema": {
            "type": "object", "properties": {"ok": {"type": "boolean"}, "inseridos": {"type": "integer"}}
        }},
        400: {"description": "Payload inválido"}
    }
})
def post_leituras_batch():
    """
    Insere 1 linha por leitura na tabela `leituras` (modelo relacional).
    A coluna `codigo_sensor` em `sensores` é usada para resolver o `sensor_id`.
    """
    dados = request.get_json(force=True)
    if not isinstance(dados, list) or not dados:
        return jsonify({"error": "Envie uma lista de leituras"}), 400

    desconhecidos = []
    preparadas = []
    for d in dados:
        try:
            codigo = (d.get("codigo_sensor") or "").strip()
            if not codigo:
                desconhecidos.append({"codigo_sensor": codigo or None, "motivo": "vazio"})
                continue
            sid = sensor_id_por_codigo(codigo)
            ts = d.get("ts") or datetime.now(timezone.utc).isoformat()
            raw = json.dumps(d.get("raw_data", {}), ensure_ascii=False)
            preparadas.append((sid, float(d["valor"]), ts, raw))
        except Exception as e:
            desconhecidos.append({"codigo_sensor": d.get("codigo_sensor"), "motivo": str(e)})

    if desconhecidos:
        return jsonify({
            "ok": False,
            "erro": "Existem sensores não cadastrados",
            "detalhes": desconhecidos
        }), 400

    inseridos = 0
    with db() as conn:
        with conn.cursor() as cur:
            for sid, valor, ts, raw in preparadas:
                cur.execute("""
                    INSERT INTO leituras (sensor_id, valor, timestamp_leitura, status_leitura, raw_data, created_at)
                    VALUES (%s, %s, %s, 'normal', %s, NOW(6))
                """, (sid, valor, ts, raw))
                inseridos += 1

    return jsonify({"ok": True, "inseridos": inseridos}), 201

# ========= ML: FEATURES =========
@app.get("/ml/features")
@swag_from({
    "tags": ["ML"],
    "responses": {200: {"description": "Features esperadas pelo modelo", "schema": {
        "type": "object",
        "properties": {"features": {"type": "array", "items": {"type": "string"}}}
    }}}
})
def ml_features():
    try:
        _, features, info = load_model_if_needed()
        return jsonify({"features": features, "info": {"umbral": info.get("umidade_solo_threshold")}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 503

# ========= ML: PREDICT (payload) =========
@app.post("/ml/predict")
@swag_from({
    "tags": ["ML"],
    "parameters": [{
        "name": "body", "in": "body", "required": True,
        "schema": {
            "oneOf": [
                {"type": "object"},
                {"type": "array", "items": {"type": "object"}}
            ],
            "description": "Objeto ou lista de objetos contendo as features usadas no treino (veja /ml/features)."
        }
    }],
    "responses": {
        200: {"description": "OK"},
        503: {"description": "Modelo indisponível"}
    }
})
def ml_predict():
    try:
        model, features, _ = load_model_if_needed()
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    payload = request.get_json(force=True)
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        return jsonify({"error": "Envie um objeto JSON ou lista de objetos."}), 400

    resultados = predict_df(df)
    return jsonify({
        "features": features,
        "n_amostras": len(df),
        "resultados": resultados
    }), 200

# ========= ML: PREDICT com última leitura do banco =========
@app.get("/ml/predict/now")
@swag_from({
    "tags": ["ML"],
    "responses": {
        200: {"description": "Predição com a última leitura de cada sensor"},
        404: {"description": "Leituras insuficientes"},
        503: {"description": "Modelo indisponível"}
    }
})
def ml_predict_now():
    try:
        _, features, _ = load_model_if_needed()
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    # Pega último valor de cada sensor_id no banco
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT l.sensor_id, l.valor
                FROM leituras l
                JOIN (
                    SELECT sensor_id, MAX(timestamp_leitura) AS ts
                    FROM leituras
                    GROUP BY sensor_id
                ) m ON l.sensor_id = m.sensor_id AND l.timestamp_leitura = m.ts
                WHERE l.sensor_id IN (%s)
            """ % (", ".join(["%s"] * len(SENSOR_MAP))), tuple(SENSOR_MAP.keys()))
            rows = cur.fetchall()

    if not rows:
        return jsonify({"error": "Sem leituras recentes"}), 404

    # Monta dict por feature
    sample = {name: None for name in features}
    for r in rows:
        sid = int(r["sensor_id"])
        feature = SENSOR_MAP.get(sid)
        if feature in sample:  # usa só as colunas do modelo
            try:
                sample[feature] = float(r["valor"])
            except Exception:
                sample[feature] = None

    df = pd.DataFrame([sample])
    resultados = predict_df(df)
    return jsonify({
        "features": features,
        "amostra": sample,
        "resultados": resultados
    }), 200

# ========= MAIN =========
if __name__ == "__main__":
    # Dica: para produção, use um servidor WSGI (gunicorn/uwsgi). Aqui é só dev:
    app.run(host="0.0.0.0", port=5000, debug=True)
