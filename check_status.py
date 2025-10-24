#!/usr/bin/env python3
"""
check_status.py

Script de diagnóstico para verificar el estado del proyecto TelcoVision.
Revisa: Git, DVC, MLflow, archivos requeridos, y configuración.

Uso:
python check_status.py
"""

import os
import sys
from pathlib import Path
import subprocess
from typing import Tuple, List


class Colors:
    """Colores ANSI para terminal."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Imprime un encabezado."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_check(passed: bool, message: str, details: str = ""):
    """Imprime resultado de un chequeo."""
    icon = f"{Colors.GREEN}✅" if passed else f"{Colors.RED}❌"
    print(f"{icon} {message}{Colors.END}")
    if details:
        print(f"   {Colors.YELLOW}{details}{Colors.END}")


def run_command(cmd: List[str]) -> Tuple[bool, str]:
    """Ejecuta un comando y retorna (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, str(e)


def check_git_status():
    """Verifica el estado de Git."""
    print_header("GIT STATUS")
    
    # Git instalado
    success, _ = run_command(["git", "--version"])
    print_check(success, "Git instalado")
    
    if not success:
        return
    
    # Repo inicializado
    git_dir = Path(".git")
    print_check(git_dir.exists(), "Repositorio Git inicializado")
    
    if not git_dir.exists():
        return
    
    # Remote configurado
    success, output = run_command(["git", "remote", "-v"])
    has_remote = bool(output)
    print_check(has_remote, "Remote configurado", output.split('\n')[0] if has_remote else "")
    
    # Branch actual
    success, branch = run_command(["git", "branch", "--show-current"])
    if success:
        print_check(True, f"Branch actual: {branch}")
    
    # Archivos sin commitear
    success, output = run_command(["git", "status", "--short"])
    uncommitted = bool(output)
    if uncommitted:
        print_check(False, "Hay cambios sin commitear", f"{len(output.split(chr(10)))} archivos")
    else:
        print_check(True, "Working directory limpio")


def check_dvc_status():
    """Verifica el estado de DVC."""
    print_header("DVC STATUS")
    
    # DVC instalado
    success, _ = run_command(["dvc", "version"])
    print_check(success, "DVC instalado")
    
    if not success:
        return
    
    # DVC inicializado
    dvc_dir = Path(".dvc")
    print_check(dvc_dir.exists(), "DVC inicializado")
    
    if not dvc_dir.exists():
        return
    
    # Remote configurado
    success, output = run_command(["dvc", "remote", "list"])
    has_remote = bool(output)
    print_check(has_remote, "DVC remote configurado", output if has_remote else "")
    
    # Archivos .dvc
    dvc_files = list(Path(".").rglob("*.dvc"))
    print_check(len(dvc_files) > 0, f"Archivos versionados: {len(dvc_files)}")
    for dvc_file in dvc_files[:5]:  # Mostrar primeros 5
        print(f"   - {dvc_file}")
    if len(dvc_files) > 5:
        print(f"   ... y {len(dvc_files) - 5} más")
    
    # Pipeline DVC
    dvc_yaml = Path("dvc.yaml")
    print_check(dvc_yaml.exists(), "Pipeline DVC (dvc.yaml)")
    
    if dvc_yaml.exists():
        # Ver stages
        success, output = run_command(["dvc", "dag"])
        if success:
            print(f"   Pipeline stages:")
            for line in output.split('\n')[:10]:
                print(f"   {line}")


def check_mlflow_status():
    """Verifica la configuración de MLflow."""
    print_header("MLFLOW STATUS")
    
    # MLflow instalado
    try:
        import mlflow
        print_check(True, f"MLflow instalado (v{mlflow.__version__})")
    except ImportError:
        print_check(False, "MLflow NO instalado")
        return
    
    # Tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        print_check(True, "MLFLOW_TRACKING_URI configurado", tracking_uri)
        is_remote = tracking_uri.startswith("http")
        print_check(is_remote, "Modo: REMOTO (DagsHub)" if is_remote else "Modo: LOCAL")
    else:
        print_check(True, "MLFLOW_TRACKING_URI: LOCAL (default)")
    
    # Experimento
    experiment = os.getenv("MLFLOW_EXPERIMENT", "default")
    print_check(True, f"Experimento: {experiment}")
    
    # Directorio mlruns
    mlruns_dir = Path("mlruns")
    if mlruns_dir.exists():
        experiments = list(mlruns_dir.iterdir())
        print_check(True, f"Directorio mlruns/ encontrado ({len(experiments)} experimentos)")
    
    # Credenciales (si es remoto)
    if tracking_uri.startswith("http"):
        username = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        print_check(bool(username), "MLFLOW_TRACKING_USERNAME configurado")
        print_check(bool(password), "MLFLOW_TRACKING_PASSWORD configurado")


def check_project_structure():
    """Verifica la estructura del proyecto."""
    print_header("PROJECT STRUCTURE")
    
    required_files = {
        "README.md": "Documentación principal",
        "requirements.txt": "Dependencias Python",
        "params.yaml": "Parámetros del pipeline",
        "dvc.yaml": "Pipeline DVC",
        ".gitignore": "Exclusiones Git",
        ".env.example": "Template de configuración",
        "src/data_prep.py": "Script de limpieza",
        "src/train.py": "Script de entrenamiento",
    }
    
    for file_path, description in required_files.items():
        exists = Path(file_path).exists()
        print_check(exists, f"{file_path}", description)
    
    # Directorios
    print("\n")
    required_dirs = [
        "data/raw",
        "data/processed",
        "models",
        "src",
        "params_experiments"
    ]
    
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        print_check(exists, f"{dir_path}/")


def check_data_files():
    """Verifica los archivos de datos."""
    print_header("DATA FILES")
    
    data_files = {
        "data/raw/telco_churn.csv": "Dataset raw",
        "data/raw/telco_churn.csv.dvc": "DVC tracking (raw)",
        "data/processed/telco_churn_processed.csv": "Dataset procesado",
        "data/processed/telco_churn_processed.csv.dvc": "DVC tracking (processed)",
    }
    
    for file_path, description in data_files.items():
        path = Path(file_path)
        exists = path.exists()
        size = f"({path.stat().st_size // 1024} KB)" if exists else ""
        print_check(exists, f"{file_path}", f"{description} {size}")


def check_environment():
    """Verifica el entorno Python."""
    print_header("PYTHON ENVIRONMENT")
    
    # Python version
    print_check(True, f"Python {sys.version.split()[0]}")
    
    # Paquetes críticos
    packages = [
        "mlflow",
        "dvc",
        "pandas",
        "sklearn",
        "joblib",
        "yaml",
    ]
    
    for package in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print_check(True, f"{package} ({version})")
        except ImportError:
            print_check(False, f"{package} NO instalado")


def check_experiments():
    """Verifica configuraciones de experimentos."""
    print_header("EXPERIMENTS CONFIGS")
    
    params_dir = Path("params_experiments")
    
    if not params_dir.exists():
        print_check(False, "Directorio params_experiments/ no encontrado")
        return
    
    yaml_files = list(params_dir.glob("*.yaml")) + list(params_dir.glob("*.yml"))
    
    print_check(len(yaml_files) > 0, f"Configuraciones encontradas: {len(yaml_files)}")
    
    for yaml_file in yaml_files:
        print(f"   - {yaml_file.name}")


def generate_summary():
    """Genera un resumen final."""
    print_header("SUMMARY")
    
    checks = {
        "Git": Path(".git").exists(),
        "DVC": Path(".dvc").exists(),
        "MLflow": True,  # Ya verificado antes
        "Pipeline DVC": Path("dvc.yaml").exists(),
        "Datos raw": Path("data/raw/telco_churn.csv").exists(),
        "Scripts": Path("src/train.py").exists(),
        "Configs experimentos": len(list(Path("params_experiments").glob("*.yaml"))) > 0 if Path("params_experiments").exists() else False,
    }
    
    passed = sum(checks.values())
    total = len(checks)
    percentage = (passed / total) * 100
    
    for name, status in checks.items():
        print_check(status, name)
    
    print(f"\n{Colors.BOLD}Completitud del proyecto: {passed}/{total} ({percentage:.0f}%){Colors.END}")
    
    if percentage == 100:
        print(f"\n{Colors.GREEN}🎉 ¡Proyecto completamente configurado!{Colors.END}")
    elif percentage >= 75:
        print(f"\n{Colors.YELLOW}⚠️  Proyecto casi listo. Completa los checks faltantes.{Colors.END}")
    else:
        print(f"\n{Colors.RED}❌ Proyecto incompleto. Revisa el README.md{Colors.END}")


def main():
    """Función principal."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                   TELCOVISION - PROJECT STATUS CHECK                   ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    print(Colors.END)
    
    check_git_status()
    check_dvc_status()
    check_mlflow_status()
    check_project_structure()
    check_data_files()
    check_environment()
    check_experiments()
    generate_summary()
    
    print(f"\n{Colors.BLUE}{'='*80}{Colors.END}\n")


if __name__ == "__main__":
    main()
