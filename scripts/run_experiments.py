"""
run_experiments.py

Script para ejecutar múltiples experimentos con diferentes configuraciones de hiperparámetros.

Funcionalidad:
1. Lee configuraciones desde params_experiments/*.yaml
2. Ejecuta train.py para cada configuración
3. Registra cada experimento en MLflow con tags descriptivos
4. Genera reporte comparativo al final

Uso:
python scripts/run_experiments.py --configs params_experiments/ --experiment telcovision_experiments

Estructura esperada de params_experiments/:
    params_experiments/
    ├── exp1_rf_baseline.yaml
    ├── exp2_rf_tuned.yaml
    ├── exp3_logistic.yaml
    └── ...
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml
import pandas as pd


def load_experiment_configs(configs_dir: Path) -> List[Dict[str, Any]]:
    """Carga todas las configuraciones .yaml del directorio."""
    configs = []
    
    if not configs_dir.exists():
        print(f"❌ Directorio no encontrado: {configs_dir}")
        return configs
    
    yaml_files = sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))
    
    if not yaml_files:
        print(f"⚠️  No se encontraron archivos .yaml en {configs_dir}")
        return configs
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                config["_config_file"] = yaml_file.name
                configs.append(config)
                print(f"✅ Cargado: {yaml_file.name}")
        except Exception as e:
            print(f"❌ Error cargando {yaml_file.name}: {e}")
    
    return configs


def run_training(
    config_path: Path,
    experiment_name: str,
    use_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta train.py con una configuración específica.
    
    Returns:
        Diccionario con métricas del experimento
    """
    cmd = [
        sys.executable,
        "src/train.py",
        "--params", str(config_path)
    ]
    
    if not use_mlflow:
        cmd.append("--no-mlflow")
    
    # Configurar variables de entorno
    env = os.environ.copy()
    env["MLFLOW_EXPERIMENT"] = experiment_name
    
    # ✅ ASEGURAR: Si no hay MLFLOW_TRACKING_URI, usar local
    if "MLFLOW_TRACKING_URI" not in env or not env["MLFLOW_TRACKING_URI"]:
        env.pop("MLFLOW_TRACKING_URI", None)  # Remover si existe vacío
    
    print(f"\n🚀 Ejecutando: {' '.join(cmd)}")
    print(f"   Experimento: {experiment_name}")
    print(f"   Config: {config_path.name}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Entrenamiento completado")
        print(result.stdout)
        
        # Intentar leer métricas del archivo
        metrics_path = Path("models/metrics.json")
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                return {
                    "config": config_path.name,
                    "status": "success",
                    **metrics
                }
        
        return {
            "config": config_path.name,
            "status": "success",
            "metrics_file": "not_found"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en entrenamiento:")
        print(e.stderr)
        return {
            "config": config_path.name,
            "status": "failed",
            "error": str(e)
        }


def generate_report(results: List[Dict[str, Any]], output_path: Path):
    """Genera un reporte comparativo de los experimentos."""
    if not results:
        print("⚠️  No hay resultados para generar reporte")
        return
    
    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(results)
    
    # Ordenar por mejor métrica (por ejemplo, roc_auc)
    if "roc_auc" in df.columns:
        df = df.sort_values("roc_auc", ascending=False)
    
    # Guardar reporte
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Reporte CSV guardado: {csv_path}")
    
    # JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Reporte JSON guardado: {json_path}")
    
    # Mostrar resumen en consola
    print("\n" + "="*80)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*80)
    
    # Seleccionar columnas relevantes para mostrar
    display_cols = ["config", "status"]
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    display_cols.extend([c for c in metric_cols if c in df.columns])
    
    if display_cols:
        print(df[display_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))
    
    print("="*80)
    
    # Mejor modelo
    if "roc_auc" in df.columns and not df["roc_auc"].isna().all():
        best_idx = df["roc_auc"].idxmax()
        best_config = df.loc[best_idx, "config"]
        best_score = df.loc[best_idx, "roc_auc"]
        print(f"\n🏆 Mejor modelo: {best_config} (ROC-AUC: {best_score:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Ejecutar múltiples experimentos de ML")
    parser.add_argument(
        "--configs",
        default="params_experiments/",
        help="Directorio con archivos de configuración .yaml"
    )
    parser.add_argument(
        "--experiment",
        default="telcovision_experiments",
        help="Nombre del experimento en MLflow"
    )
    parser.add_argument(
        "--report",
        default="reports/experiments_comparison.csv",
        help="Ruta para guardar el reporte comparativo"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Desactivar MLflow tracking"
    )
    
    args = parser.parse_args()
    
    configs_dir = Path(args.configs)
    
    print("="*80)
    print("EJECUTOR DE EXPERIMENTOS - TelcoVision")
    print("="*80)
    print(f"Directorio de configs: {configs_dir}")
    print(f"Experimento MLflow: {args.experiment}")
    print(f"Reporte: {args.report}")
    print("="*80)
    
    # Cargar configuraciones
    configs = load_experiment_configs(configs_dir)
    
    if not configs:
        print("\n❌ No se encontraron configuraciones para ejecutar")
        return 1
    
    print(f"\n📋 Se ejecutarán {len(configs)} experimentos\n")
    
    # Ejecutar experimentos
    results = []
    for i, config in enumerate(configs, 1):
        config_file = config.get("_config_file", f"config_{i}.yaml")
        config_path = configs_dir / config_file
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENTO {i}/{len(configs)}: {config_file}")
        print(f"{'='*80}")
        
        result = run_training(
            config_path=config_path,
            experiment_name=args.experiment,
            use_mlflow=not args.no_mlflow
        )
        
        results.append(result)
    
    # Generar reporte
    print(f"\n{'='*80}")
    print("GENERANDO REPORTE FINAL")
    print(f"{'='*80}")
    
    generate_report(results, Path(args.report))
    
    # Resumen final
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    print(f"\n✨ Ejecución completada:")
    print(f"   Total experimentos: {len(results)}")
    print(f"   Exitosos: {successful}")
    print(f"   Fallidos: {failed}")
    
    if not args.no_mlflow:
        print(f"\n💡 Revisa los experimentos en MLflow UI:")
        print(f"   mlflow ui --port 5000")
        print(f"   http://localhost:5000")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
