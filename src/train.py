"""
train.py

Entrenador de modelo para TelcoVision.
- Lee parámetros desde `params.yaml` (o CLI)
- Usa el dataset limpio (`data/processed/telco_churn_processed.csv`)
- Entrena un modelo base (LogisticRegression o RandomForest, según params.yaml)
- Calcula métricas: accuracy, precision, recall, f1, roc_auc
- Guarda el modelo en `models/model.joblib` y las métricas en `models/metrics.json`
- Registra todo en MLflow (local o remoto según configuración)

Uso:
python src/train.py --params params.yaml
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
from sklearn.linear_model import LogisticRegression
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
    """
    Verifica si MLflow está disponible y debe ser usado.
    
    Funciona tanto en modo LOCAL como REMOTO.
    - Si MLFLOW_TRACKING_URI está vacío/no definido → tracking LOCAL (./mlruns/)
    - Si MLFLOW_TRACKING_URI tiene URL → tracking REMOTO (DagsHub, etc.)
    """
    if no_mlflow_flag:
        print("[INFO] MLflow desactivado por flag --no-mlflow")
        return False
    
    try:
        import importlib.util
        spec = importlib.util.find_spec("mlflow")
        
        if spec is None:
            print("[WARN] MLflow no está instalado. pip install mlflow")
            return False
        
        print("[INFO] MLflow disponible y habilitado")
        return True
        
    except Exception as e:
        print(f"[WARN] Error verificando MLflow: {e}")
        return False


def resolve_config(params: Dict[str, Any], cli: argparse.Namespace) -> Dict[str, Any]:
    """
    Combina params.yaml con overrides de CLI y devuelve un dict normalizado.
    Compatible con estructura simplificada de params.yaml.
    """
    # Obtener paths (puede estar directamente en root o en sección paths)
    paths = params.get("paths", {})
    if not paths:
        # Si no hay sección paths, usar valores directos
        paths = {
            "processed_data": params.get("processed_data", "data/processed/telco_churn_processed.csv"),
            "model_path": params.get("model_path", "models/model.joblib"),
            "metrics_path": params.get("metrics_path", "models/metrics.json")
        }
    
    # Split params (puede estar en sección split o directamente en root)
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    
    # Model config
    model_cfg = params.get("model", {})
    
    # Target
    target = params.get("target", "churn")

    cfg = {
        "input_path": Path(cli.input) if cli.input else Path(paths.get("processed_data", "")),
        "model_path": Path(cli.out) if cli.out else Path(paths.get("model_path", "models/model.joblib")),
        "metrics_path": Path(cli.metrics) if cli.metrics else Path(paths.get("metrics_path", "models/metrics.json")),
        "target": cli.target or target,
        "test_size": cli.test_size if cli.test_size is not None else float(test_size),
        "random_state": cli.random_state if cli.random_state is not None else int(random_state),
        "model_cfg": model_cfg,
    }
    
    if not cfg["input_path"] or not str(cfg["input_path"]):
        raise ValueError("No se definió la ruta de datos procesados (paths.processed_data o --input).")
    
    return cfg


# ---------- Modelo ----------

def build_pipeline_from_params(model_cfg: Dict[str, Any], random_state: int) -> Pipeline:
    """Crea Pipeline(StandardScaler -> Modelo) con parámetros desde params."""
    mtype = model_cfg.get("type", "LogisticRegression")
    params = (model_cfg.get("parameters", {}) or {}).copy()
    params.setdefault("random_state", random_state)

    if mtype == "RandomForest":
        params.setdefault("n_estimators", 100)
        params.setdefault("n_jobs", -1)
        model = RandomForestClassifier(**params)
    elif mtype == "LogisticRegression":
        params.setdefault("max_iter", 200)
        model = LogisticRegression(**params)
    else:
        raise ValueError("Sólo se soporta model.type=RandomForest o LogisticRegression.")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
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

    print(f"\n{'='*80}")
    print("INICIO DE ENTRENAMIENTO")
    print(f"{'='*80}")
    print(f"Dataset: {inp}")
    print(f"Modelo: {model_cfg.get('type', 'LogisticRegression')}")
    print(f"Target: {target}")
    print(f"Test size: {test_size}")
    print(f"Random state: {random_state}")
    
    df = pd.read_csv(inp)
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no está en el dataset procesado.")

    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"Shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Distribución target: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    model = build_pipeline_from_params(model_cfg, random_state)

    if use_mlflow:
        import mlflow
        
        # Configurar tracking URI explícitamente
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"[MLFLOW] Tracking URI: {tracking_uri} (REMOTO)")
        else:
            # Si está vacío, MLflow usa ./mlruns/ por defecto
            print(f"[MLFLOW] Tracking URI: ./mlruns/ (LOCAL)")
        
        # Configurar experimento
        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "telcovision_experiments")
        mlflow.set_experiment(experiment_name)
        print(f"[MLFLOW] Experimento: {experiment_name}")
        
        with mlflow.start_run() as run:
            print(f"[MLFLOW] Run ID: {run.info.run_id}")
            
            # Log de parámetros del modelo
            model_params = model_cfg.get("parameters", {})
            if model_params:
                mlflow.log_params(model_params)
            
            # Log de parámetros adicionales
            mlflow.log_param("model_type", model_cfg.get("type", "LogisticRegression"))
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("target", target)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples_train", X_train.shape[0])
            mlflow.log_param("n_samples_test", X_test.shape[0])

            # Entrenar
            print("\n[TRAIN] Entrenando modelo...")
            model.fit(X_train, y_train)
            print("[TRAIN] OK - Entrenamiento completado")
            
            # Evaluar
            print("[EVAL] Calculando métricas...")
            metrics = evaluate(model, X_test, y_test)
            print("[EVAL] OK - Métricas calculadas")

            # Log de métricas
            mlflow.log_metrics(metrics)
            print("[MLFLOW] OK - Métricas registradas")

            # Guardar modelo localmente
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            print(f"[SAVE] Modelo guardado: {model_path}")
            
            # Log del modelo como artefacto en MLflow
            mlflow.log_artifact(str(model_path))
            print("[MLFLOW] OK - Modelo registrado como artefacto")

            # Guardar métricas localmente
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[SAVE] Métricas guardadas: {metrics_path}")
            
            # Log de métricas como artefacto
            mlflow.log_artifact(str(metrics_path))
            print("[MLFLOW] OK - Métricas registradas como artefacto")
            
            # EXTRA: Intentar registrar el modelo en el Model Registry
            try:
                model_name = "TelcoChurn_Model"
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    registered_model_name=model_name
                )
                print(f"[MLFLOW] OK - Modelo registrado en Model Registry: {model_name}")
            except Exception as e:
                print(f"[MLFLOW] WARN -  No se pudo registrar en Model Registry: {e}")
                print("[MLFLOW] (Esto es normal en algunos servidores remotos)")
            
            print(f"\n[MLFLOW] Run completado: {run.info.run_id}")
            print(f"[MLFLOW] Ver en UI: http://localhost:5000 (si es local)")
            
    else:
        print("\n[INFO] MLflow desactivado - entrenamiento sin tracking")
        
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"[SAVE] Modelo guardado: {model_path}")

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVE] Métricas guardadas: {metrics_path}")

    return model, metrics


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Entrenamiento RandomForest con params.yaml")
    ap.add_argument("--params", default="params.yaml", help="Ruta al params.yaml")
    ap.add_argument("--input", help="Ruta CSV procesado (override de params.paths.processed_data)")
    ap.add_argument("--out", help="Ruta de salida del modelo .joblib (override de params.paths.model_path)")
    ap.add_argument("--metrics", help="Ruta de salida de métricas .json (override de params.paths.metrics_path)")
    ap.add_argument("--target", help="Columna objetivo (override de params.target; default: churn)")
    ap.add_argument("--test-size", type=float, help="Tamaño del set de test (override de params.test_size)")
    ap.add_argument("--random-state", type=int, help="Semilla aleatoria (override de params.random_state)")
    ap.add_argument("--no-mlflow", action="store_true", help="Desactiva MLflow aunque esté disponible")
    return ap.parse_args()


def main():
    print(f"\n{'='*80}")
    print("TELCOVISION - ENTRENAMIENTO DE MODELO")
    print(f"{'='*80}\n")
    
    args = parse_args()
    params = load_params(Path(args.params))
    cfg = resolve_config(params, args)
    use_mlflow = mlflow_is_enabled(args.no_mlflow)

    model, metrics = train_and_save(cfg, use_mlflow)

    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"Modelo guardado en: {cfg['model_path']}")
    print(f"Métricas guardadas en: {cfg['metrics_path']}")
    print("\nMétricas:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: N/A")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
