"""
register_best_model.py

Script para registrar el mejor modelo en MLflow Model Registry.

Flujo:
1. Busca todos los runs del experimento
2. Ordena por la métrica especificada (default: roc_auc)
3. Registra el mejor run en el Model Registry con el nombre especificado

Uso:
python scripts/register_best_model.py --experiment telcovision_experiments --metric roc_auc --model-name TelcoChurn_Model

Requisitos:
- MLflow debe estar configurado (MLFLOW_TRACKING_URI en .env)
- Debe haber al menos un run completado en el experimento
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


def find_best_run(
    experiment_name: str,
    metric_name: str = "metrics.roc_auc",
    ascending: bool = False
) -> Optional[mlflow.entities.Run]:
    """
    Busca el mejor run en un experimento según una métrica.
    
    Args:
        experiment_name: Nombre del experimento
        metric_name: Nombre de la métrica (debe incluir prefijo 'metrics.')
        ascending: Si True, menor es mejor; si False, mayor es mejor
    
    Returns:
        El mejor run encontrado o None si no hay runs
    """
    client = MlflowClient()
    
    # Buscar experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experimento '{experiment_name}' no encontrado.")
        return None
    
    print(f"🔍 Buscando runs en experimento: {experiment_name} (ID: {experiment.experiment_id})")
    
    # Buscar runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if not runs:
        print(f"❌ No se encontraron runs en el experimento '{experiment_name}'")
        return None
    
    return runs[0]


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model"
) -> str:
    """
    Registra un modelo en MLflow Model Registry.
    
    Args:
        run_id: ID del run que contiene el modelo
        model_name: Nombre para el modelo en el registry
        artifact_path: Path del artefacto del modelo dentro del run
    
    Returns:
        Version del modelo registrado
    """
    client = MlflowClient()
    
    # Construir URI del modelo
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    print(f"📦 Registrando modelo desde: {model_uri}")
    
    try:
        # Registrar modelo
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"✅ Modelo registrado: {model_name} (version {model_version.version})")
        
        # Agregar descripción
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Mejor modelo basado en run {run_id}"
        )
        
        return model_version.version
        
    except Exception as e:
        print(f"❌ Error al registrar modelo: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Registrar mejor modelo en MLflow Model Registry")
    parser.add_argument(
        "--experiment",
        default="telcovision_experiments",
        help="Nombre del experimento (default: telcovision_experiments)"
    )
    parser.add_argument(
        "--metric",
        default="roc_auc",
        help="Métrica para ordenar (sin prefijo 'metrics.', default: roc_auc)"
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Ordenar de menor a mayor (por defecto: mayor a menor)"
    )
    parser.add_argument(
        "--model-name",
        default="TelcoChurn_Model",
        help="Nombre del modelo en el registry (default: TelcoChurn_Model)"
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Path del artefacto del modelo (default: model)"
    )
    parser.add_argument(
        "--run-id",
        help="Run ID específico a registrar (opcional, ignora búsqueda automática)"
    )
    
    args = parser.parse_args()
    
    # Verificar configuración de MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        print("⚠️  MLFLOW_TRACKING_URI no está configurado. Usando tracking local (./mlruns/)")
        mlflow.set_tracking_uri("file:///./mlruns")
    else:
        print(f"🌐 Usando MLflow tracking remoto: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    
    # Buscar o usar run específico
    if args.run_id:
        print(f"📌 Usando run específico: {args.run_id}")
        run_id = args.run_id
    else:
        # Buscar mejor run
        metric_name = f"metrics.{args.metric}" if not args.metric.startswith("metrics.") else args.metric
        best_run = find_best_run(args.experiment, metric_name, args.ascending)
        
        if best_run is None:
            print("\n❌ No se pudo encontrar un run válido para registrar.")
            return
        
        run_id = best_run.info.run_id
        
        # Mostrar info del mejor run
        print(f"\n✨ Mejor run encontrado:")
        print(f"   Run ID: {run_id}")
        print(f"   {args.metric}: {best_run.data.metrics.get(args.metric, 'N/A')}")
        print(f"   Otros params: {best_run.data.params}")
    
    # Registrar modelo
    try:
        version = register_model(run_id, args.model_name, args.artifact_path)
        print(f"\n🎉 Registro exitoso!")
        print(f"   Modelo: {args.model_name}")
        print(f"   Versión: {version}")
        print(f"   Run ID: {run_id}")
        
    except Exception as e:
        print(f"\n❌ Error en el registro: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
