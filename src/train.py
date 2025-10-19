"""
train.py

Entrena un modelo RandomForest dentro de un Pipeline(StandardScaler -> RandomForest)
usando parámetros definidos en params.yaml (y opcionalmente sobreescritos por CLI).

- Lee params de --params (default: params.yaml)
- Usa paths.processed_data / paths.model_path / paths.metrics_path (overrides por CLI)
- Calcula métricas: accuracy, precision, recall, f1 y (si aplica) roc_auc
- Guarda modelo (.joblib) y metrics (.json)
- (Opcional) Loguea en MLflow si el paquete está instalado y hay MLFLOW_TRACKING_URI

Uso:
python src/train.py --params params.yaml
Opcionales:
  --input   data/processed/telco_churn_processed.csv
  --out     models/model.joblib
  --metrics models/metrics.json
  --target  churn
  --test-size 0.2
  --random-state 42
  --no-mlflow  (para desactivar MLflow incluso si está disponible)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- Utilidades ----------

def load_params(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def mlflow_is_enabled(no_mlflow_flag: bool) -> bool:
    if no_mlflow_flag:
        return False
    try:
        import importlib.util
        spec = importlib.util.find_spec("mlflow")
        return spec is not None and bool(os.getenv("MLFLOW_TRACKING_URI"))
    except Exception:
        return False


def resolve_config(params: Dict[str, Any], cli: argparse.Namespace) -> Dict[str, Any]:
    """Combina params.yaml con overrides de CLI y devuelve un dict normalizado."""
    paths = params.get("paths", {}) if isinstance(params.get("paths", {}), dict) else {}
    split = params.get("split", {}) if isinstance(params.get("split", {}), dict) else {}
    model_cfg = params.get("model", {}) if isinstance(params.get("model", {}), dict) else {}

    cfg = {
        "input_path": Path(cli.input) if cli.input else Path(paths.get("processed_data", "")),
        "model_path": Path(cli.out) if cli.out else Path(paths.get("model_path", "models/model.joblib")),
        "metrics_path": Path(cli.metrics) if cli.metrics else Path(paths.get("metrics_path", "models/metrics.json")),
        "target": cli.target or params.get("target", "churn"),
        "test_size": cli.test_size if cli.test_size is not None else float(split.get("test_size", 0.2)),
        "random_state": cli.random_state if cli.random_state is not None else int(split.get("random_state", 42)),
        "model_cfg": model_cfg,
    }
    if not cfg["input_path"]:
        raise ValueError("No se definió la ruta de datos procesados (paths.processed_data o --input).")
    return cfg


# ---------- Modelo ----------

def build_pipeline_from_params(model_cfg: Dict[str, Any], random_state: int) -> Pipeline:
    """Crea Pipeline(StandardScaler -> RandomForest) con parámetros desde params."""
    mtype = model_cfg.get("type", "RandomForest")
    if mtype != "RandomForest":
        raise ValueError("Sólo se soporta model.type=RandomForest en esta versión.")

    rf_params = (model_cfg.get("parameters", {}) or {}).copy()
    # Defaults sensatos si no vinieron en params.yaml
    rf_params.setdefault("n_estimators", 200)
    rf_params.setdefault("random_state", random_state)
    rf_params.setdefault("n_jobs", -1)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(**rf_params))
    ])
    return pipe


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    # Probabilidades (si el estimador las soporta)
    try:
        proba = model.predict_proba(X_test)
    except Exception:
        proba = None

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    # roc_auc (binaria o, si se puede, multiclase macro-ovr)
    if proba is not None:
        try:
            if proba.ndim == 2 and proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            elif proba.ndim == 2 and proba.shape[1] > 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def train_and_save(cfg: Dict[str, Any], use_mlflow: bool) -> Tuple[Pipeline, Dict[str, Any]]:
    inp: Path = cfg["input_path"]
    model_path: Path = cfg["model_path"]
    metrics_path: Path = cfg["metrics_path"]
    target: str = cfg["target"]
    test_size: float = cfg["test_size"]
    random_state: int = cfg["random_state"]
    model_cfg: Dict[str, Any] = cfg["model_cfg"]

    if not inp.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {inp}")

    df = pd.read_csv(inp)
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no está en el dataset procesado.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_pipeline_from_params(model_cfg, random_state)

    if use_mlflow:
        import mlflow
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "default"))
        with mlflow.start_run():
            # log de hiperparámetros del bosque
            mlflow.log_params(model_cfg.get("parameters", {}))
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("target", target)

            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            # artefactos
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(str(metrics_path))
    else:
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return model, metrics


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Entrenamiento RandomForest con params.yaml")
    ap.add_argument("--params", default="params.yaml", help="Ruta al params.yaml")
    ap.add_argument("--input", help="Ruta CSV procesado (override de params.paths.processed_data)")
    ap.add_argument("--out", help="Ruta de salida del modelo .joblib (override de params.paths.model_path)")
    ap.add_argument("--metrics", help="Ruta de salida de métricas .json (override de params.paths.metrics_path)")
    ap.add_argument("--target", help="Columna objetivo (override de params.target; default: churn)")
    ap.add_argument("--test-size", type=float, help="Tamaño del set de test (override de params.split.test_size)")
    ap.add_argument("--random-state", type=int, help="Semilla aleatoria (override de params.split.random_state)")
    ap.add_argument("--no-mlflow", action="store_true", help="Desactiva MLflow aunque esté disponible")
    return ap.parse_args()


def main():
    args = parse_args()
    params = load_params(Path(args.params))
    cfg = resolve_config(params, args)
    use_mlflow = mlflow_is_enabled(args.no_mlflow)

    model, metrics = train_and_save(cfg, use_mlflow)

    print(f"\nModelo guardado en: {cfg['model_path']}")
    print("Métricas:")
    for k, v in metrics.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
