#!/usr/bin/env python3
"""
Predição de "pode_irrigar" usando o modelo salvo em models/model.joblib.

Modos de entrada:
  --csv caminho.csv            -> arquivo CSV com colunas de features
  --json-file caminho.json     -> arquivo JSON (lista de objetos)
  --json-stdin                 -> lê JSON da entrada padrão

Saída:
  JSON com a lista de resultados: index, classe (0/1) e probabilidade (0..1), se disponível.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"

def load_model_and_info():
    if not MODEL_PATH.exists():
        raise SystemExit("[ERRO] models/model.joblib não encontrado. Rode o treino primeiro.")
    if not INFO_PATH.exists():
        raise SystemExit("[ERRO] models/model_info.json não encontrado. Rode o treino primeiro.")
    model = load(MODEL_PATH)
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    features = info.get("features", [])
    if not features:
        raise SystemExit("[ERRO] Lista de 'features' não encontrada em model_info.json.")
    return model, features, info

def read_input(args, features):
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.json_file:
        payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        df = pd.DataFrame(payload)
    elif args.json_stdin:
        import sys
        raw = sys.stdin.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"[ERRO] JSON inválido na entrada padrão: {e}\nConteúdo lido:\n{raw}")
        df = pd.DataFrame(payload)
    else:
        raise SystemExit("Informe --csv, --json-file ou --json-stdin")

    # Mantém apenas as colunas usadas no treino
    df = df.reindex(columns=features)
    return df

def predict(model, X):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Caminho para CSV com colunas de features")
    parser.add_argument("--json-file", type=str, help="Caminho para arquivo JSON (lista de objetos)")
    parser.add_argument("--json-stdin", action="store_true", help="Ler JSON da entrada padrão")
    parser.add_argument("--print-features", dest="print_features", action="store_true",
                        help="Apenas imprime as features esperadas e sai")
    args = parser.parse_args()

    model, features, info = load_model_and_info()

    if args.print_features:
        print(json.dumps({"features": features}, ensure_ascii=False, indent=2))
        return

    X = read_input(args, features)
    resultados = predict(model, X)
    print(json.dumps({
        "features": features,
        "n_amostras": len(X),
        "resultados": resultados
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
