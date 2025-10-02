import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# ======== CONFIG BÁSICA =========
APP_TITLE = "Estufa Inteligente • KPIs e Alertas"
SOIL_MOISTURE_THRESHOLD = 30  # % para alerta
DEFAULT_WINDOW_MIN = 60       # minutos de janela
TIME_BUCKET = "30S"           # grade temporal p/ alinhar leituras

# Códigos dos sensores (ajuste se necessário)
SENSOR_CODES = {
    "temp_ar":   "DHT22_TEMP_AR_ESP32_01",
    "umid_ar":   "DHT22_UMID_AR_ESP32_01",
    "temp_solo": "LM35_TEMP_SOLO_01",
    "umid_solo": "CAP_SOLO_ESP32_01",
    "luz":       "LDR_LUX_ESP32_01",
    "co2":       "MQ135_CO2_ESP32_01",
}

# ======== CONEXÃO MYSQL FIXA ========
def _new_mysql_conn():
    import pymysql
    return pymysql.connect(
        host="192.185.217.50",
        user="qualidad_estufas",
        password="Padr@ao321",
        database="qualidad_estufas",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

# ======== CARREGAR DADOS (UTC + leitura manual) =========
@st.cache_data(ttl=10)
def load_data_db_window(minutes: int) -> pd.DataFrame:
    """Busca leituras usando o relógio UTC do MySQL e converte tipos de forma robusta."""
    minutes = int(minutes) if minutes and int(minutes) > 0 else DEFAULT_WINDOW_MIN
    sql = f"""
        SELECT
            COALESCE(s.codigo_sensor, CONCAT('sensor_', l.sensor_id)) AS codigo_sensor,
            l.valor,
            l.timestamp_leitura
        FROM qualidad_estufas.leituras l
        LEFT JOIN qualidad_estufas.sensores s ON s.id = l.sensor_id
        WHERE l.timestamp_leitura BETWEEN DATE_SUB(UTC_TIMESTAMP(), INTERVAL {minutes} MINUTE)
                                      AND UTC_TIMESTAMP()
        ORDER BY l.timestamp_leitura ASC;
    """
    conn = _new_mysql_conn()
    try:
        conn.ping(reconnect=True)
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        try: conn.close()
        except: pass

    df = pd.DataFrame(rows, columns=["codigo_sensor", "valor", "timestamp_leitura"])

    # Conversões robustas
    if "timestamp_leitura" in df.columns:
        df["timestamp_leitura"] = pd.to_datetime(df["timestamp_leitura"], errors="coerce", infer_datetime_format=True)
    if "valor" in df.columns:
        df["valor"] = df["valor"].astype(str).str.replace(",", ".", regex=False).str.strip()
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    if "codigo_sensor" in df.columns:
        df["codigo_sensor"] = df["codigo_sensor"].astype(str).str.strip()

    df = df.dropna(subset=["timestamp_leitura"])
    return df

@st.cache_data(ttl=10)
def mysql_clock_info() -> dict:
    """Mostra relógio do servidor e último timestamp gravado (debug)."""
    sql = """
        SELECT NOW() AS now_srv, UTC_TIMESTAMP() AS now_utc,
               @@time_zone AS tz, @@system_time_zone AS sys_tz
    """
    sql2 = "SELECT MAX(timestamp_leitura) AS max_ts FROM qualidad_estufas.leituras"
    conn = _new_mysql_conn()
    try:
        with conn.cursor() as c:
            c.execute(sql); a = c.fetchone()
            c.execute(sql2); b = c.fetchone()
        return {"now_srv": a["now_srv"], "now_utc": a["now_utc"], "tz": a["tz"], "sys_tz": a["sys_tz"], "max_ts": b["max_ts"]}
    finally:
        try: conn.close()
        except: pass

def pivot_by_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """Alinha as leituras numa grade temporal e faz forward-fill para preencher lacunas."""
    if df.empty:
        return df.copy()
    df = df.copy()
    df["bucket"] = df["timestamp_leitura"].dt.floor(TIME_BUCKET)
    wide = df.pivot_table(index="bucket", columns="codigo_sensor", values="valor", aggfunc="last").sort_index()
    wide = wide.ffill(limit=10)  # último valor conhecido
    wide = wide.reset_index().rename(columns={"bucket": "timestamp"})
    return wide

# ======== PREDIÇÃO (com normalização de nomes) ========
def predict_now_from_model(latest_row: pd.Series) -> dict:
    """
    Carrega o modelo e cria o vetor de entrada com os nomes esperados.
    Faz mapeamento entre nomes do dashboard e nomes do modelo.
    """
    try:
        import joblib, os
        model_path = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "model.joblib")
        model = joblib.load(model_path)
    except Exception as ex:
        return {"ok": False, "error": f"Não foi possível carregar o modelo: {ex}"}

    # 1) Descobrir quais features o modelo espera
    try:
        expected = list(getattr(model, "feature_names_in_", []))
    except Exception:
        expected = []
    # Se o modelo não tiver feature_names_in_, defina manualmente a lista abaixo:
    if not expected:
        expected = ["temperatura_solo", "qualidade_ar_ppm", "temperatura_ar", "umidade_ar", "umidade_solo", "luminosidade"]

    # 2) Mapeamento de nomes do dashboard -> nomes esperados pelo modelo
    #    (ajuste aqui se seu modelo usar outras labels)
    dash_to_model = {
        "temp_ar":   "temperatura_ar",
        "umid_ar":   "umidade_ar",
        "temp_solo": "temperatura_solo",
        "umid_solo": "umidade_solo",
        "luz":       "luminosidade",
        "co2":       "qualidade_ar_ppm",
    }

    # latest_row contém colunas no padrão do dashboard; convertemos:
    latest_dict = {}
    for dash_col, model_col in dash_to_model.items():
        if dash_col in latest_row.index and pd.notnull(latest_row[dash_col]):
            latest_dict[model_col] = float(latest_row[dash_col])

    # 3) Montar X com as colunas exatamente na ordem esperada
    missing = [col for col in expected if col not in latest_dict]
    if missing:
        return {"ok": False, "error": f"columns are missing: {missing}"}

    import numpy as np
    X = pd.DataFrame([[latest_dict[c] for c in expected]], columns=expected)

    # 4) Predizer
    try:
        yhat = model.predict(X)[0]
        score = None
        try:
            proba = model.predict_proba(X)[0]
            score = float(max(proba))
        except Exception:
            pass
        return {"ok": True, "pred": int(yhat), "score": score}
    except Exception as ex:
        return {"ok": False, "error": f"Falha na predição: {ex}"}

# ======== UI ========
st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")
st.title(APP_TITLE)

colA, colB, colC = st.columns([2,1,1])
with colB:
    window_min = st.number_input("Janela (minutos)", min_value=5, max_value=24*60, value=DEFAULT_WINDOW_MIN, step=5)
with colC:
    st.write("Threshold umid. solo (%)")
    th = st.slider(" ", min_value=5, max_value=100, value=int(SOIL_MOISTURE_THRESHOLD), step=1, label_visibility="collapsed")

# Carregar dados (UTC) + fallbacks se necessário
df_raw = load_data_db_window(window_min)
if df_raw.empty:
    df_raw = load_data_db_window(360)
if df_raw.empty:
    df_raw = load_data_db_window(1440)


wide = pivot_by_sensor(df_raw)
if wide.empty:
    st.warning("Sem dados na janela selecionada.")
    st.stop()

# renomear p/ nomes amigáveis no dashboard
rename_map = {
    SENSOR_CODES["temp_ar"]:   "temp_ar",
    SENSOR_CODES["umid_ar"]:   "umid_ar",
    SENSOR_CODES["temp_solo"]: "temp_solo",
    SENSOR_CODES["umid_solo"]: "umid_solo",
    SENSOR_CODES["luz"]:       "luz",
    SENSOR_CODES["co2"]:       "co2",
}
wide = wide.rename(columns=rename_map)

# KPIs (linha já forward-filled)
latest = wide.iloc[-1]
kpi_cols = st.columns(6)
def kpi(c, label, val, unit="", avg=None):
    if pd.notnull(val):
        c.metric(label, f"{float(val):.2f}{unit}", f"média {avg:.2f}{unit}" if avg is not None else None)
    else:
        c.metric(label, "—", None)

kpi_map = {
    "temp_ar":  ("Temp. Ar", "°C"),
    "umid_ar":  ("Umid. Ar", "%"),
    "temp_solo":("Temp. Solo", "°C"),
    "umid_solo":("Umid. Solo", "%"),
    "luz":      ("Luminosidade", ""),
    "co2":      ("Qualidade Ar (ppm)", "ppm"),
}
for i, (col, (label, unit)) in enumerate(kpi_map.items()):
    if col in wide.columns:
        kpi(kpi_cols[i], label, latest[col], unit, wide[col].mean())

# ALERTA (usa valor ffill do último instante)
alert_placeholder = st.empty()
if "umid_solo" in wide.columns:
    last_umid = latest["umid_solo"]
    if pd.notnull(last_umid) and float(last_umid) < th:
        alert_placeholder.error(f"⚠️ Umidade do solo baixa ({float(last_umid):.1f}%) < {th}% — **Irrigação necessária**")
    elif pd.notnull(last_umid):
        alert_placeholder.success(f"✅ Umidade do solo OK ({float(last_umid):.1f}%) ≥ {th}%")

# ======== GRÁFICOS COM TÍTULO ========
st.subheader("Séries temporais")
gcol1, gcol2 = st.columns(2)

with gcol1:
    if "umid_solo" in wide.columns:
        st.markdown("**Umidade do Solo (%)**")
        st.line_chart(wide.set_index("timestamp")[["umid_solo"]], y_label="Umidade (%)")
    if "temp_ar" in wide.columns:
        st.markdown("**Temperatura do Ar (°C)**")
        st.line_chart(wide.set_index("timestamp")[["temp_ar"]], y_label="°C")

with gcol2:
    if "umid_ar" in wide.columns:
        st.markdown("**Umidade do Ar (%)**")
        st.line_chart(wide.set_index("timestamp")[["umid_ar"]], y_label="Umidade (%)")
    if "luz" in wide.columns:
        st.markdown("**Luminosidade (LDR)**")
        st.line_chart(wide.set_index("timestamp")[["luz"]], y_label="Luminosidade")

# ======== PREDIÇÃO (se houver modelo salvo) ========
st.subheader("Predição (modelo ML)")
use_model = st.toggle("Usar modelo salvo (ml/models/model.joblib) para prever irrigação agora", value=True)
if use_model:
    # vetor com nomes do dashboard
    dash_features = ["temp_ar","umid_ar","temp_solo","umid_solo","luz","co2"]
    have = [f for f in dash_features if f in wide.columns]
    res = predict_now_from_model(latest[have])
    if res.get("ok"):
        y = res["pred"]; score = res.get("score")
        st.markdown("**Resultado:** 🚿 **Irrigar** (classe 1)" if y == 1 else "**Resultado:** 💧 **Não irrigar** (classe 0)")
        if score is not None: st.caption(f"Confiança: {score:.2f}")
    else:
        st.warning(f"Falha na predição: {res.get('error','erro desconhecido')}")

with st.expander("Dados brutos (após alinhamento/ffill)"):
    st.dataframe(wide.tail(200), use_container_width=True)
