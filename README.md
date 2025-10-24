# 📊 TelcoVision - MLOps End-to-End con MLflow + DVC + DagsHub

Sistema completo de predicción de churn de clientes de telecomunicaciones con versionado de datos (DVC), experimentos (MLflow) y colaboración (DagsHub).

---

## 🎯 Objetivo del Proyecto

Construir un pipeline end-to-end de Machine Learning para predecir el churn de clientes, implementando:
- ✅ Versionado de datos con **DVC**
- ✅ Tracking de experimentos con **MLflow**
- ✅ Pipelines reproducibles con **DVC pipelines**
- ✅ Colaboración remota con **DagsHub**
- ✅ Model Registry para gestión de modelos

---

## 📁 Estructura del Proyecto

```
telcovision/
├── .dvc/                         # Configuración de DVC
├── .github/workflows/            # CI/CD (opcional)
├── data/
│   ├── raw/                      # Dataset original (versionado con DVC)
│   │   └── telco_churn.csv
│   └── processed/                # Dataset limpio (versionado con DVC)
│       └── telco_churn_processed.csv
├── models/                       # Modelos entrenados (versionado con DVC)
│   ├── model.joblib
│   └── metrics.json
├── mlruns/                       # Experimentos MLflow (local, NO subir a Git)
├── params_experiments/           # Configuraciones de experimentos
│   ├── exp1_rf_baseline.yaml
│   ├── exp2_rf_optimized.yaml
│   ├── exp3_rf_regularized.yaml
│   ├── exp4_logistic_baseline.yaml
│   └── exp5_logistic_l1.yaml
├── reports/                      # Reportes comparativos
│   └── experiments_comparison.csv
├── scripts/
│   ├── register_best_model.py   # Registro en Model Registry
│   └── run_experiments.py        # Ejecutor de múltiples experimentos
├── src/
│   ├── data_prep.py             # Limpieza de datos
│   └── train.py                  # Entrenamiento con MLflow
├── .env                          # Variables de entorno (NO subir a Git)
├── .env.example                  # Plantilla de configuración
├── .gitignore                    # Exclusiones de Git
├── .dvcignore                    # Exclusiones de DVC
├── dvc.yaml                      # Pipeline DVC
├── dvc.lock                      # Lock file del pipeline
├── params.yaml                   # Parámetros del pipeline
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

---

## 🚀 Setup Inicial (Etapa 1)

### 1.1 Requisitos Previos

- Python 3.9+
- Git
- Cuenta en [DagsHub](https://dagshub.com) (opcional para remoto)

### 1.2 Clonar o Crear Repositorio

**Opción A: Proyecto nuevo**
```bash
mkdir telcovision
cd telcovision
git init
```

**Opción B: Clonar desde GitHub**
```bash
git clone https://github.com/<TU_USER>/telcovision.git
cd telcovision
```

### 1.3 Crear Entorno Virtual

```bash
# Con conda
conda create -n telcovision python=3.9 -y
conda activate telcovision

# O con venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 1.4 Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 1.5 Configurar Variables de Entorno

```bash
# Copiar plantilla
cp .env.example .env

# Editar .env con tus credenciales
# Para empezar en LOCAL, deja MLFLOW_TRACKING_URI vacío
```

**Contenido de `.env` para LOCAL:**
```bash
MLFLOW_EXPERIMENT=telcovision_experiments
# MLFLOW_TRACKING_URI=  # Vacío para tracking local
```

### 1.6 Inicializar DVC

```bash
# Inicializar DVC (si no está inicializado)
dvc init

# Agregar dataset raw
dvc add data/raw/telco_churn.csv

# Commit cambios
git add data/raw/.gitignore data/raw/telco_churn.csv.dvc .dvc/
git commit -m "feat: add raw dataset with DVC"
```

**✅ Entregable Etapa 1:**
- Repositorio con estructura base
- Dataset raw versionado con DVC
- Entorno configurado

---

## 🧹 Limpieza y Features (Etapa 2)

### 2.1 Ejecutar Limpieza de Datos

```bash
# Opción A: Con DVC pipeline (recomendado)
dvc repro data_prep

# Opción B: Directamente
python src/data_prep.py --input data/raw/telco_churn.csv --out data/processed/telco_churn_processed.csv
```

### 2.2 Versionar Dataset Procesado

```bash
# Agregar dataset procesado a DVC
dvc add data/processed/telco_churn_processed.csv

# Commit cambios
git add data/processed/.gitignore data/processed/telco_churn_processed.csv.dvc dvc.lock
git commit -m "feat: add processed dataset with DVC"
```

### 2.3 Verificar Pipeline DVC

```bash
# Ver status del pipeline
dvc status

# Ver DAG del pipeline
dvc dag
```

**✅ Entregable Etapa 2:**
- Pipeline reproducible en `dvc.yaml`
- Dataset crudo y limpio versionados
- `dvc.lock` actualizado

---

## 🤖 Entrenamiento de Modelo (Etapa 3)

### 3.1 Entrenar Modelo Base con MLflow (Local)

**Terminal 1: Levantar MLflow UI**
```bash
mlflow ui --port 5000
```

Abre tu navegador en: http://localhost:5000

**Terminal 2: Entrenar modelo**
```bash
# Con configuración por defecto (params.yaml)
python src/train.py --params params.yaml

# O con DVC
dvc repro train
```

### 3.2 Verificar Experimento en MLflow UI

1. Abre http://localhost:5000
2. Ve a "Experiments" → "telcovision_experiments"
3. Revisa métricas: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
4. Descarga artefactos si es necesario

### 3.3 Modificar Hiperparámetros

Edita `params.yaml` para cambiar configuración:

```yaml
model:
  type: RandomForest  # O LogisticRegression
  parameters:
    n_estimators: 200  # Cambiar valores
    max_depth: 15
    # ... otros parámetros
```

Luego re-entrena:
```bash
dvc repro train
```

**✅ Entregable Etapa 3:**
- Modelo entrenado con `params.yaml`
- Métricas en `models/metrics.json`
- Pipeline DVC actualizado
- Al menos 1 run registrado en MLflow

---

## 🧪 Experimentos (Etapa 4)

### 4.1 Ejecutar Múltiples Experimentos

Usa el script `run_experiments.py` para correr todas las configuraciones en `params_experiments/`:

```bash
python scripts/run_experiments.py \
  --configs params_experiments/ \
  --experiment telcovision_experiments \
  --report reports/experiments_comparison.csv
```

Esto ejecutará 5 experimentos:
1. **exp1_rf_baseline**: Random Forest conservador
2. **exp2_rf_optimized**: Random Forest agresivo
3. **exp3_rf_regularized**: Random Forest regularizado
4. **exp4_logistic_baseline**: Logistic Regression base
5. **exp5_logistic_l1**: Logistic Regression con L1

### 4.2 Comparar Experimentos

**En MLflow UI:**
1. Selecciona múltiples runs
2. Click en "Compare"
3. Analiza gráficos de métricas
4. Identifica mejor modelo

**En CSV:**
```bash
# Ver reporte comparativo
cat reports/experiments_comparison.csv
```

### 4.3 Registrar Mejor Modelo en Model Registry

```bash
# Busca automáticamente el mejor run por roc_auc
python scripts/register_best_model.py \
  --experiment telcovision_experiments \
  --metric roc_auc \
  --model-name TelcoChurn_Model

# O especifica un run_id manualmente
python scripts/register_best_model.py \
  --run-id <RUN_ID> \
  --model-name TelcoChurn_Model
```

### 4.4 Verificar Model Registry

En MLflow UI → "Models" → "TelcoChurn_Model" → Ver versiones

**✅ Entregable Etapa 4:**
- Al menos 3 corridas con diferentes hiperparámetros
- Reporte comparativo en `reports/experiments_comparison.csv`
- Mejor modelo registrado en MLflow Model Registry
- Screenshots de comparación en MLflow UI

---

## 🌐 Publicar en DagsHub (Opcional)

### 5.1 Crear Repositorio en DagsHub

1. Ve a https://dagshub.com
2. Click "New Repository"
3. Nombra: `telcovision`
4. Marca "Initialize with README" (opcional)

### 5.2 Conectar Git con DagsHub

```bash
# Si es repo nuevo
git remote add origin https://dagshub.com/<TU_USER>/telcovision.git

# Si ya tenías origin en GitHub, usa otro nombre
git remote add dagshub https://dagshub.com/<TU_USER>/telcovision.git

# Push código
git push -u origin main  # o dagshub main
```

### 5.3 Configurar DVC Remote (DagsHub)

```bash
# Agregar remote DVC
dvc remote add origin https://dagshub.com/<TU_USER>/telcovision.dvc

# Configurar autenticación (local)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <TU_USER>
dvc remote modify origin --local password <DAGSHUB_TOKEN>

# Push datos y modelos
dvc push -r origin
```

> **Nota:** Genera tu token en: https://dagshub.com/user/settings/tokens

### 5.4 Configurar MLflow Tracking Remoto (DagsHub)

Edita `.env`:
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<TU_USER>/telcovision.mlflow
MLFLOW_TRACKING_USERNAME=<TU_USER>
MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>
MLFLOW_EXPERIMENT=telcovision_experiments
```

Luego re-entrena:
```bash
python src/train.py --params params.yaml
```

### 5.5 Verificar en DagsHub

1. **Code**: Archivos versionados con Git
2. **Data**: Datasets versionados con DVC
3. **Experiments**: Runs de MLflow
4. **Models**: Artefactos y modelos

---

## 📊 Comandos Útiles

### DVC
```bash
# Ver status del pipeline
dvc status

# Reproducir pipeline completo
dvc repro

# Ver gráfico de dependencias
dvc dag

# Pull datos/modelos desde remoto
dvc pull -r origin

# Push datos/modelos a remoto
dvc push -r origin
```

### MLflow
```bash
# Levantar UI local
mlflow ui --port 5000

# Ver experimentos
mlflow experiments list

# Ver runs de un experimento
mlflow runs list --experiment-name telcovision_experiments
```

### Git
```bash
# Ver status
git status

# Agregar archivos
git add .

# Commit
git commit -m "mensaje"

# Push a remoto
git push origin main
```

---

## 🔧 Troubleshooting

### Problema: MLflow no registra runs

**Solución:**
1. Verifica que `MLFLOW_TRACKING_URI` esté configurado
2. Si usas local, asegúrate que la carpeta `mlruns/` exista
3. Verifica que MLflow esté instalado: `pip list | grep mlflow`

### Problema: DVC no encuentra archivos

**Solución:**
```bash
# Re-inicializar DVC
dvc checkout

# Pull datos desde remoto
dvc pull -r origin
```

### Problema: Error de autenticación en DagsHub

**Solución:**
1. Genera nuevo token en: https://dagshub.com/user/settings/tokens
2. Actualiza `.env` con el nuevo token
3. Re-configura remote DVC:
```bash
dvc remote modify origin --local password <NUEVO_TOKEN>
```

---

## 📚 Recursos Adicionales

- [Documentación MLflow](https://mlflow.org/docs/latest/index.html)
- [Documentación DVC](https://dvc.org/doc)
- [Tutoriales DagsHub](https://dagshub.com/docs)
- [Scikit-learn](https://scikit-learn.org/)

---

## 📝 Notas del Proyecto

### Dataset
- **Fuente**: Telco Customer Churn
- **Registros**: 10,000
- **Features**: 12 (9 categóricas, 3 numéricas)
- **Target**: `churn` (0: No, 1: Sí)
- **Desbalance**: ~27% churn

### Modelos Probados
- Random Forest (baseline, optimized, regularized)
- Logistic Regression (L2, L1)

### Mejores Resultados
Ver `reports/experiments_comparison.csv` para comparación completa.

---

## 🤝 Contribuciones

Este es un proyecto educativo. Pull requests son bienvenidos.

---

## 📄 Licencia

MIT License

---

## 👤 Autor

Creado como proyecto integrador de MLOps.

---

## 🎓 Etapas Completadas

- [x] **Etapa 1**: Setup inicial con Git + DagsHub + DVC
- [x] **Etapa 2**: Limpieza de datos y pipeline reproducible
- [x] **Etapa 3**: Entrenamiento con MLflow y métricas
- [x] **Etapa 4**: Múltiples experimentos y Model Registry

---

**¡Happy MLOps! 🚀**
