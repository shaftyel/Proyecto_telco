#!/usr/bin/env python3
"""
setup_project.py

Script de setup automático para TelcoVision.
Verifica dependencias, crea estructura de carpetas y configura el proyecto.

Uso:
python setup_project.py
"""

import os
import sys
from pathlib import Path
import subprocess


def print_section(title: str):
    """Imprime una sección visual."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def check_command(cmd: str) -> bool:
    """Verifica si un comando está disponible."""
    try:
        subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_dependencies():
    """Verifica que las dependencias del sistema estén instaladas."""
    print_section("VERIFICANDO DEPENDENCIAS DEL SISTEMA")
    
    dependencies = {
        "python": "Python 3.9+",
        "git": "Git",
        "pip": "pip"
    }
    
    missing = []
    
    for cmd, name in dependencies.items():
        if check_command(cmd):
            print(f"✅ {name} instalado")
        else:
            print(f"❌ {name} NO encontrado")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Instala las siguientes herramientas antes de continuar:")
        for tool in missing:
            print(f"   - {tool}")
        return False
    
    print("\n✅ Todas las dependencias del sistema están instaladas")
    return True


def create_directory_structure():
    """Crea la estructura de directorios del proyecto."""
    print_section("CREANDO ESTRUCTURA DE DIRECTORIOS")
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "mlruns",
        "params_experiments",
        "reports",
        "scripts",
        "src",
        ".github/workflows"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Creado: {dir_path}")
    
    print("\n✅ Estructura de directorios completa")


def create_gitignore_files():
    """Crea archivos .gitignore en carpetas específicas."""
    print_section("CREANDO ARCHIVOS .gitignore")
    
    gitignores = {
        "data/processed/.gitignore": "/telco_churn_processed.csv\n",
        "models/.gitignore": "*.joblib\n*.pkl\n*.h5\nmetrics.json\n",
        "mlruns/.gitignore": "*\n",
    }
    
    for path, content in gitignores.items():
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        print(f"✅ Creado: {path}")
    
    print("\n✅ Archivos .gitignore creados")


def check_python_packages():
    """Verifica que los paquetes Python estén instalados."""
    print_section("VERIFICANDO PAQUETES PYTHON")
    
    required_packages = [
        "mlflow",
        "dvc",
        "pandas",
        "scikit-learn",
        "joblib",
        "pyyaml",
        "dagshub"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} instalado")
        except ImportError:
            print(f"❌ {package} NO instalado")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Instala los paquetes faltantes con:")
        print(f"   pip install {' '.join(missing)}")
        print(f"\n   O instala todo desde requirements.txt:")
        print(f"   pip install -r requirements.txt")
        return False
    
    print("\n✅ Todos los paquetes Python están instalados")
    return True


def initialize_dvc():
    """Inicializa DVC si no está inicializado."""
    print_section("INICIALIZANDO DVC")
    
    dvc_dir = Path(".dvc")
    
    if dvc_dir.exists():
        print("✅ DVC ya está inicializado")
        return True
    
    try:
        subprocess.run(["dvc", "init"], check=True)
        print("✅ DVC inicializado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al inicializar DVC: {e}")
        return False


def check_env_file():
    """Verifica que exista el archivo .env."""
    print_section("VERIFICANDO CONFIGURACIÓN (.env)")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ Archivo .env encontrado")
        return True
    
    if env_example.exists():
        print("⚠️  Archivo .env no encontrado")
        print("   Copia .env.example a .env:")
        print("   cp .env.example .env")
        print("   Luego edita .env con tus credenciales")
        return False
    
    print("❌ Ni .env ni .env.example encontrados")
    return False


def print_next_steps():
    """Imprime los siguientes pasos para el usuario."""
    print_section("✨ SETUP COMPLETADO - SIGUIENTES PASOS")
    
    steps = [
        "1. Configura tus variables de entorno:",
        "   cp .env.example .env",
        "   # Edita .env con tu información",
        "",
        "2. Coloca tu dataset en data/raw/:",
        "   # Asegúrate de tener telco_churn.csv",
        "",
        "3. Versiona el dataset con DVC:",
        "   dvc add data/raw/telco_churn.csv",
        "   git add data/raw/.gitignore data/raw/telco_churn.csv.dvc",
        "   git commit -m 'Add raw dataset with DVC'",
        "",
        "4. Ejecuta el pipeline de limpieza:",
        "   dvc repro data_prep",
        "",
        "5. Entrena tu primer modelo:",
        "   # Terminal 1: mlflow ui --port 5000",
        "   # Terminal 2: python src/train.py --params params.yaml",
        "",
        "6. Ejecuta múltiples experimentos:",
        "   python scripts/run_experiments.py",
        "",
        "📚 Lee el README.md para instrucciones detalladas",
    ]
    
    for step in steps:
        print(step)


def main():
    """Función principal."""
    print_section("🚀 SETUP AUTOMÁTICO - TELCOVISION")
    
    # Verificar dependencias del sistema
    if not check_dependencies():
        print("\n❌ Setup abortado: instala las dependencias faltantes")
        return 1
    
    # Crear estructura de directorios
    create_directory_structure()
    
    # Crear .gitignore files
    create_gitignore_files()
    
    # Verificar paquetes Python
    if not check_python_packages():
        print("\n⚠️  Continúa cuando instales los paquetes Python")
    
    # Inicializar DVC
    initialize_dvc()
    
    # Verificar .env
    check_env_file()
    
    # Mostrar siguientes pasos
    print_next_steps()
    
    print("\n✅ Setup completado exitosamente!")
    print("👉 Lee el README.md para comenzar\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
