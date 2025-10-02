# -*- coding: utf-8 -*-
"""
Treino do modelo de irrigação
- Extrai dados do MySQL (UTC)
- Faz pivot + alinhamento temporal
- Gera rótulo (classe) a partir de regra simples (umidade do solo < limiar)
- Treina RandomForest
- Salva métricas (txt), matriz de confusão (png), importâncias de features, série temporal, histogramas e modelo (joblib)

Estruturas geradas (relativas à pasta atual):
- models/model.joblib
- figuras/matriz_confusao.png
- figuras/feature_importance.png
- figuras/timeseries.png
- figuras/features_hist.png
- (opcional, se houver duas classes no teste)
  - figuras/roc_curve.png
  - figuras/pr_curve.png
- reports/metrics.txt
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pymysql
import matplotlib
matplotlib.use("Agg")  # backend não interativo para salvar PNG
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

# ----------------- CONFIGS -----------------
DB_CFG = dict(
    host="192.185.217.50",
    user="qualidad_estufas",
    password="Padr@ao321",
    database="qualidad_estufas",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

# códigos dos sensores no banco -> colunas “amigáveis”
SENSOR_CODES = {
    "DHT22_TEMP_AR_ESP32_01": "temp_ar",
    "DHT22_UMID_AR_ESP32_01": "umid_ar",
    "LM35_TEMP_SOLO_01": "temp_solo",
    "CAP_SOLO_ESP32_01": "umid_solo",
    "LDR_LUX_ESP32_01": "luz",
    "MQ135_CO2_ESP32_01": "co2",
}

# Mapeamento p/ nomes esperados pelo modelo
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

TIME_BUCKET = "30s"  # usar "s" minúsculo para evitar FutureWarning
UMID_SOLO_LIMIAR = 30.0  # rótulo: irrigar = umidade_solo < 30

OUT_MODEL = os.path.join("models", "model.joblib")
OUT_METRICS = os.path.join("reports", "metrics.txt")
OUT_CM = os.path.join("figuras", "matriz_confusao.png")
OUT_FI = os.path.join("figuras", "feature_importance.png")
OUT_TS = os.path.join("figuras", "timeseries.png")
OUT_HIST = os.path.join("figuras", "features_hist.png")
OUT_ROC = os.path.join("figuras", "roc_curve.png")
OUT_PR = os.path.join("figuras", "pr_curve.png")


def _debug(msg: str):
    print(f"[DEBUG] {msg}")


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figuras", exist_ok=True)
    _debug(
        f"Pastas prontas | models: {os.path.abspath('models')} | reports: {os.path.abspath('reports')} | figuras: {os.path.abspath('figuras')}"
    )


def load_from_db(minutes: int = 24 * 60) -> pd.DataFrame:
    """Extrai leituras da janela (UTC) e retorna DataFrame bruto."""
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
    df["valor"] = df["valor"].astype(str).str.replace(",", ".", regex=False).str.strip()
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["codigo_sensor"] = df["codigo_sensor"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp_leitura"])
    return df


def pivot_align(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot por sensor e alinha em grade temporal com forward-fill."""
    if df_raw.empty:
        return df_raw

    # manter apenas os sensores de interesse
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

    # renomear colunas de código -> amigáveis
    wide = wide.rename(columns=SENSOR_CODES)

    # alinha valores faltantes (até 10 buckets)
    wide = wide.ffill(limit=10)
    wide = wide.reset_index().rename(columns={"bucket": "timestamp"})
    return wide


def build_dataset(wide: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas com nomes esperados pelo modelo e o rótulo binário."""
    if wide.empty:
        return wide

    # cria colunas do modelo (caso alguma não exista)
    for dash_name, model_name in DASH_TO_MODEL.items():
        if dash_name in wide.columns and model_name not in wide.columns:
            wide[model_name] = wide[dash_name]
        elif model_name not in wide.columns:
            wide[model_name] = np.nan

    # rótulo (classe) a partir da regra — irrigar (1) se umidade_solo < limiar
    if "umid_solo" in wide.columns:
        wide["target_irrigar"] = (wide["umid_solo"] < UMID_SOLO_LIMIAR).astype(int)
    else:
        wide["target_irrigar"] = np.nan

    # remove linhas sem target ou com features faltando
    cols_needed = FEATURES + ["target_irrigar"]
    present_cols = [c for c in cols_needed if c in wide.columns]
    ds = wide[["timestamp"] + present_cols].copy()
    ds = ds.dropna(subset=["target_irrigar"])  # precisa de target
    ds = ds.dropna(subset=FEATURES, how="any")  # precisa de todas as features

    return ds


def safe_train_test_split(X, y, test_size=0.25, random_state=42):
    """Faz o split com estratificação somente se houver pelo menos 2 amostras em cada classe.
    Se houver apenas uma classe no dataset, lança erro claro.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    if len(unique) < 2:
        raise ValueError(
            f"Dataset tem apenas uma classe (y={unique.tolist()}). Não é possível treinar classificador."
        )

    if min(counts) < 2:
        print(
            f"[AVISO] Classe minoritária com < 2 amostras {class_counts}. Fazendo split sem stratify."
        )
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=None
        )

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_and_evaluate(ds: pd.DataFrame, wide: pd.DataFrame | None = None):
    X = ds[FEATURES].values
    y = ds["target_irrigar"].values.astype(int)

    X_train, X_test, y_train, y_test = safe_train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("Accuracy:", acc)
    print("Classification Report:", report)

    # salvar métricas
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write(f"Treinado em: {datetime.utcnow().isoformat()}Z")
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in dict(zip(unique, counts)).items()}
        f.write(f"Distribuição de classes (total={len(y)}): {dist}")
        f.write(f"Accuracy: {acc:.4f}")
        f.write(report)

    # --- Matriz de confusão (sempre salva alguma imagem) ---
    out_cm_abs = os.path.abspath(OUT_CM)
    _debug(f"Gerando imagem em: {out_cm_abs}")
    try:
        if len(np.unique(y_test)) > 1:
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            plt.title("Matriz de Confusão - Modelo Irrigação")
            plt.tight_layout()
            plt.savefig(out_cm_abs, dpi=150)
            plt.close()

            # Curvas ROC e PR (apenas com duas classes)
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
                # ROC
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT_ROC, dpi=150)
                plt.close()

                # PR
                prec, rec, _ = precision_recall_curve(y_test, y_score)
                ap = average_precision_score(y_test, y_score)
                plt.figure(figsize=(5, 4))
                plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision–Recall Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT_PR, dpi=150)
                plt.close()
        else:
            # Figura informativa para classe única
            cls_vals = np.unique(y_test)
            cls = int(cls_vals[0]) if len(cls_vals) == 1 else -1
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('off')
            ax.text(0.5, 0.62, "Sem matriz de confusão", ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.45, f"Conjunto de teste possui uma única classe: {cls}", ha='center', va='center', fontsize=11)
            ax.text(0.5, 0.30, "Imagem gerada para consistência do pipeline.", ha='center', va='center', fontsize=10)
            plt.tight_layout()
            plt.savefig(out_cm_abs, dpi=150)
            plt.close()

            with open(OUT_METRICS, "a", encoding="utf-8") as f:
                f.write("Observação: conjunto de teste possui uma única classe; imagem informativa salva em figuras/matriz_confusao.png.")
    finally:
        _debug(f"Imagem criada? {os.path.exists(out_cm_abs)} -> {out_cm_abs}")

    # --- Feature Importance ---
    try:
        importances = clf.feature_importances_
        fi = list(zip(FEATURES, importances))
        fi.sort(key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(7, 4))
        plt.bar([k for k, _ in fi], [v for _, v in fi])
        plt.xticks(rotation=30, ha='right')
        plt.ylabel("Importância (Gini)")
        plt.title("Importância das Features — RandomForest")
        plt.tight_layout()
        plt.savefig(OUT_FI, dpi=150)
        plt.close()
        _debug(f"Feature importance salva em: {os.path.abspath(OUT_FI)}")
    except Exception as e:
        _debug(f"Falha ao gerar feature importance: {e}")

    # --- Série temporal (opcional) ---
    try:
        if wide is not None and not wide.empty:
            cols_plot = [c for c in ["temp_ar","umid_ar","temp_solo","umid_solo","luz","co2"] if c in wide.columns]
            if cols_plot:
                plt.figure(figsize=(10, 4.5))
                for c in cols_plot:
                    plt.plot(wide["timestamp"], wide[c], label=c)
                plt.legend(loc="best")
                plt.xlabel("Tempo (UTC)")
                plt.title("Séries temporais — Leituras de sensores")
                plt.tight_layout()
                plt.savefig(OUT_TS, dpi=150)
                plt.close()
                _debug(f"Série temporal salva em: {os.path.abspath(OUT_TS)}")
    except Exception as e:
        _debug(f"Falha ao gerar série temporal: {e}")

    # --- Histogramas das features ---
    try:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(FEATURES, 1):
            if col in ds.columns:
                plt.subplot(2, 3, i)
                plt.hist(ds[col].dropna().values, bins=20)
                plt.title(col)
        plt.tight_layout()
        plt.savefig(OUT_HIST, dpi=150)
        plt.close()
        _debug(f"Histogramas salvos em: {os.path.abspath(OUT_HIST)}")
    except Exception as e:
        _debug(f"Falha ao gerar histogramas: {e}")

    # salvar modelo
    joblib.dump(clf, OUT_MODEL)
    print("Artefatos salvos:")
    print(" -", OUT_MODEL)
    print(" -", OUT_METRICS)
    if os.path.exists(OUT_CM):
        print(" -", OUT_CM)
    if os.path.exists(OUT_FI):
        print(" -", OUT_FI)
    if os.path.exists(OUT_TS):
        print(" -", OUT_TS)
    if os.path.exists(OUT_HIST):
        print(" -", OUT_HIST)
    if os.path.exists(OUT_ROC):
        print(" -", OUT_ROC)
    if os.path.exists(OUT_PR):
        print(" -", OUT_PR)


def main():
    ensure_dirs()
    print("1) Carregando dados do banco...")
    df_raw = load_from_db(minutes=24 * 60)  # 24h
    print("Linhas brutas:", len(df_raw))

    print("2) Pivot + alinhamento temporal...")
    wide = pivot_align(df_raw)
    print("Linhas após pivot:", len(wide))

    print("3) Preparando dataset com features + target...")
    ds = build_dataset(wide)
    print("Linhas no dataset:", len(ds))

    if ds.empty:
        raise SystemExit("Dataset vazio: verifique se há leituras e colunas necessárias.")

    print("4) Treinando e avaliando...")
    train_and_evaluate(ds, wide=wide)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Treino do modelo de irrigação
- Extrai dados do MySQL (UTC)
- Faz pivot + alinhamento temporal
- Gera rótulo (classe) a partir de regra simples (umidade do solo < limiar)
- Treina RandomForest
- Salva métricas (txt), matriz de confusão (png), importâncias de features, série temporal, histogramas e modelo (joblib)

Estruturas geradas (relativas à pasta atual):
- models/model.joblib
- figuras/matriz_confusao.png
- figuras/feature_importance.png
- figuras/timeseries.png
- figuras/features_hist.png
- (opcional, se houver duas classes no teste)
  - figuras/roc_curve.png
  - figuras/pr_curve.png
- reports/metrics.txt
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pymysql
import matplotlib
matplotlib.use("Agg")  # backend não interativo para salvar PNG
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

# ----------------- CONFIGS -----------------
DB_CFG = dict(
    host="192.185.217.50",
    user="qualidad_estufas",
    password="Padr@ao321",
    database="qualidad_estufas",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

# códigos dos sensores no banco -> colunas “amigáveis”
SENSOR_CODES = {
    "DHT22_TEMP_AR_ESP32_01": "temp_ar",
    "DHT22_UMID_AR_ESP32_01": "umid_ar",
    "LM35_TEMP_SOLO_01": "temp_solo",
    "CAP_SOLO_ESP32_01": "umid_solo",
    "LDR_LUX_ESP32_01": "luz",
    "MQ135_CO2_ESP32_01": "co2",
}

# Mapeamento p/ nomes esperados pelo modelo
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

TIME_BUCKET = "30s"  # usar "s" minúsculo para evitar FutureWarning
UMID_SOLO_LIMIAR = 30.0  # rótulo: irrigar = umidade_solo < 30

OUT_MODEL = os.path.join("models", "model.joblib")
OUT_METRICS = os.path.join("reports", "metrics.txt")
OUT_CM = os.path.join("figuras", "matriz_confusao.png")
OUT_FI = os.path.join("figuras", "feature_importance.png")
OUT_TS = os.path.join("figuras", "timeseries.png")
OUT_HIST = os.path.join("figuras", "features_hist.png")
OUT_ROC = os.path.join("figuras", "roc_curve.png")
OUT_PR = os.path.join("figuras", "pr_curve.png")


def _debug(msg: str):
    print(f"[DEBUG] {msg}")


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figuras", exist_ok=True)
    _debug(
        f"Pastas prontas | models: {os.path.abspath('models')} | reports: {os.path.abspath('reports')} | figuras: {os.path.abspath('figuras')}"
    )


def load_from_db(minutes: int = 24 * 60) -> pd.DataFrame:
    """Extrai leituras da janela (UTC) e retorna DataFrame bruto."""
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
    df["valor"] = df["valor"].astype(str).str.replace(",", ".", regex=False).str.strip()
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["codigo_sensor"] = df["codigo_sensor"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp_leitura"])
    return df


def pivot_align(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot por sensor e alinha em grade temporal com forward-fill."""
    if df_raw.empty:
        return df_raw

    # manter apenas os sensores de interesse
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

    # renomear colunas de código -> amigáveis
    wide = wide.rename(columns=SENSOR_CODES)

    # alinha valores faltantes (até 10 buckets)
    wide = wide.ffill(limit=10)
    wide = wide.reset_index().rename(columns={"bucket": "timestamp"})
    return wide


def build_dataset(wide: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas com nomes esperados pelo modelo e o rótulo binário."""
    if wide.empty:
        return wide

    # cria colunas do modelo (caso alguma não exista)
    for dash_name, model_name in DASH_TO_MODEL.items():
        if dash_name in wide.columns and model_name not in wide.columns:
            wide[model_name] = wide[dash_name]
        elif model_name not in wide.columns:
            wide[model_name] = np.nan

    # rótulo (classe) a partir da regra — irrigar (1) se umidade_solo < limiar
    if "umid_solo" in wide.columns:
        wide["target_irrigar"] = (wide["umid_solo"] < UMID_SOLO_LIMIAR).astype(int)
    else:
        wide["target_irrigar"] = np.nan

    # remove linhas sem target ou com features faltando
    cols_needed = FEATURES + ["target_irrigar"]
    present_cols = [c for c in cols_needed if c in wide.columns]
    ds = wide[["timestamp"] + present_cols].copy()
    ds = ds.dropna(subset=["target_irrigar"])  # precisa de target
    ds = ds.dropna(subset=FEATURES, how="any")  # precisa de todas as features

    return ds


def safe_train_test_split(X, y, test_size=0.25, random_state=42):
    """Faz o split com estratificação somente se houver pelo menos 2 amostras em cada classe.
    Se houver apenas uma classe no dataset, lança erro claro.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    if len(unique) < 2:
        raise ValueError(
            f"Dataset tem apenas uma classe (y={unique.tolist()}). Não é possível treinar classificador."
        )

    if min(counts) < 2:
        print(
            f"[AVISO] Classe minoritária com < 2 amostras {class_counts}. Fazendo split sem stratify."
        )
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=None
        )

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_and_evaluate(ds: pd.DataFrame, wide: pd.DataFrame | None = None):
    X = ds[FEATURES].values
    y = ds["target_irrigar"].values.astype(int)

    X_train, X_test, y_train, y_test = safe_train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("Accuracy:", acc)
    print("Classification Report:", report)

    # salvar métricas
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write(f"Treinado em: {datetime.utcnow().isoformat()}Z")
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in dict(zip(unique, counts)).items()}
        f.write(f"Distribuição de classes (total={len(y)}): {dist}")
        f.write(f"Accuracy: {acc:.4f}")
        f.write(report)

    # --- Matriz de confusão (sempre salva alguma imagem) ---
    out_cm_abs = os.path.abspath(OUT_CM)
    _debug(f"Gerando imagem em: {out_cm_abs}")
    try:
        if len(np.unique(y_test)) > 1:
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            plt.title("Matriz de Confusão - Modelo Irrigação")
            plt.tight_layout()
            plt.savefig(out_cm_abs, dpi=150)
            plt.close()

            # Curvas ROC e PR (apenas com duas classes)
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
                # ROC
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT_ROC, dpi=150)
                plt.close()

                # PR
                prec, rec, _ = precision_recall_curve(y_test, y_score)
                ap = average_precision_score(y_test, y_score)
                plt.figure(figsize=(5, 4))
                plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision–Recall Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT_PR, dpi=150)
                plt.close()
        else:
            # Figura informativa para classe única
            cls_vals = np.unique(y_test)
            cls = int(cls_vals[0]) if len(cls_vals) == 1 else -1
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('off')
            ax.text(0.5, 0.62, "Sem matriz de confusão", ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.45, f"Conjunto de teste possui uma única classe: {cls}", ha='center', va='center', fontsize=11)
            ax.text(0.5, 0.30, "Imagem gerada para consistência do pipeline.", ha='center', va='center', fontsize=10)
            plt.tight_layout()
            plt.savefig(out_cm_abs, dpi=150)
            plt.close()

            with open(OUT_METRICS, "a", encoding="utf-8") as f:
                f.write("Observação: conjunto de teste possui uma única classe; imagem informativa salva em figuras/matriz_confusao.png.")
    finally:
        _debug(f"Imagem criada? {os.path.exists(out_cm_abs)} -> {out_cm_abs}")

    # --- Feature Importance ---
    try:
        importances = clf.feature_importances_
        fi = list(zip(FEATURES, importances))
        fi.sort(key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(7, 4))
        plt.bar([k for k, _ in fi], [v for _, v in fi])
        plt.xticks(rotation=30, ha='right')
        plt.ylabel("Importância (Gini)")
        plt.title("Importância das Features — RandomForest")
        plt.tight_layout()
        plt.savefig(OUT_FI, dpi=150)
        plt.close()
        _debug(f"Feature importance salva em: {os.path.abspath(OUT_FI)}")
    except Exception as e:
        _debug(f"Falha ao gerar feature importance: {e}")

    # --- Série temporal (opcional) ---
    try:
        if wide is not None and not wide.empty:
            cols_plot = [c for c in ["temp_ar","umid_ar","temp_solo","umid_solo","luz","co2"] if c in wide.columns]
            if cols_plot:
                plt.figure(figsize=(10, 4.5))
                for c in cols_plot:
                    plt.plot(wide["timestamp"], wide[c], label=c)
                plt.legend(loc="best")
                plt.xlabel("Tempo (UTC)")
                plt.title("Séries temporais — Leituras de sensores")
                plt.tight_layout()
                plt.savefig(OUT_TS, dpi=150)
                plt.close()
                _debug(f"Série temporal salva em: {os.path.abspath(OUT_TS)}")
    except Exception as e:
        _debug(f"Falha ao gerar série temporal: {e}")

    # --- Histogramas das features ---
    try:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(FEATURES, 1):
            if col in ds.columns:
                plt.subplot(2, 3, i)
                plt.hist(ds[col].dropna().values, bins=20)
                plt.title(col)
        plt.tight_layout()
        plt.savefig(OUT_HIST, dpi=150)
        plt.close()
        _debug(f"Histogramas salvos em: {os.path.abspath(OUT_HIST)}")
    except Exception as e:
        _debug(f"Falha ao gerar histogramas: {e}")

    # salvar modelo
    joblib.dump(clf, OUT_MODEL)
    print("Artefatos salvos:")
    print(" -", OUT_MODEL)
    print(" -", OUT_METRICS)
    if os.path.exists(OUT_CM):
        print(" -", OUT_CM)
    if os.path.exists(OUT_FI):
        print(" -", OUT_FI)
    if os.path.exists(OUT_TS):
        print(" -", OUT_TS)
    if os.path.exists(OUT_HIST):
        print(" -", OUT_HIST)
    if os.path.exists(OUT_ROC):
        print(" -", OUT_ROC)
    if os.path.exists(OUT_PR):
        print(" -", OUT_PR)


def main():
    ensure_dirs()
    print("1) Carregando dados do banco...")
    df_raw = load_from_db(minutes=24 * 60)  # 24h
    print("Linhas brutas:", len(df_raw))

    print("2) Pivot + alinhamento temporal...")
    wide = pivot_align(df_raw)
    print("Linhas após pivot:", len(wide))

    print("3) Preparando dataset com features + target...")
    ds = build_dataset(wide)
    print("Linhas no dataset:", len(ds))

    if ds.empty:
        raise SystemExit("Dataset vazio: verifique se há leituras e colunas necessárias.")

    print("4) Treinando e avaliando...")
    train_and_evaluate(ds, wide=wide)


if __name__ == "__main__":
    main()
