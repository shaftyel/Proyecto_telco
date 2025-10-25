# 📊 TelcoVision - Sistema MLOps para Predicción de Churn

**Trabajo Práctico Integrador - Laboratorio de Minería de Datos**

Sistema completo de Machine Learning con versionado de datos, tracking de experimentos y despliegue colaborativo para predecir la cancelación de servicios (churn) en clientes de telecomunicaciones.

---

## 👥 Equipo

- **Estudiantes:** 
				Evelyn Solange Irusta / 
				Ignacio Heck
- **Institución:** ISTEA
- **Materia:** Laboratorio de Minería de Datos
- **Fecha:** Octubre 2025

---

## 🎯 Objetivo

Implementar un pipeline end-to-end de Machine Learning con:
- Versionado de datos y modelos (DVC)
- Tracking de experimentos (MLflow)
- Colaboración remota (DagsHub)
- Reproducibilidad total del proyecto

---

## 📁 Estructura del Proyecto

```
Proyecto_Laboratorio_MineriaV2/
├── data/
│   ├── raw/                           # Dataset original (10,000 registros)
│   │   └── telco_churn.csv           # Versionado con DVC
│   └── processed/                     # Dataset limpio
│       └── telco_churn_processed.csv # Versionado con DVC
├── models/
│   ├── model.joblib                  # Modelo entrenado (RandomForest)
│   └── metrics.json                  # Métricas de evaluación
├── params_experiments/               # 5 configuraciones de experimentos
│   ├── exp1_rf_baseline.yaml
│   ├── exp2_rf_optimized.yaml
│   ├── exp3_rf_regularized.yaml
│   ├── exp4_logistic_baseline.yaml
│   └── exp5_logistic_l1.yaml
├── reports/
│   ├── experiments_comparison.csv    # Comparación de todos los experimentos
│   └── experiments_comparison.json
├── scripts/
│   ├── run_experiments.py           # Ejecutor de múltiples experimentos
│   └── register_best_model.py       # Registro en Model Registry
├── src/
│   ├── data_prep.py                 # Limpieza y transformación de datos
│   └── train.py                      # Entrenamiento con MLflow
├── .dvc/                             # Configuración DVC
├── mlruns/                           # Experimentos MLflow (local)
├── params.yaml                       # Configuración principal del modelo
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```

---

## 🔧 Tecnologías Utilizadas

| Herramienta | Propósito | Versión |
|-------------|-----------|---------|
| **Python** | Lenguaje principal | 3.9+ |
| **scikit-learn** | Modelos ML | 1.3.0 |
| **pandas** | Manipulación de datos | 2.0.3 |
| **MLflow** | Tracking de experimentos | 2.15.0 |
| **DVC** | Versionado de datos/modelos | 3.50.0 |
| **DagsHub** | Colaboración y hosting remoto | - |
| **Git** | Control de versiones | 2.x |

---

## 📊 Dataset

### Características
- **Nombre:** Telco Customer Churn Dataset
- **Registros:** 10,000 clientes
- **Features:** 24 (después de one-hot encoding)
  - **Originales:** 12 features (9 categóricas, 3 numéricas)
  - **Procesadas:** 24 features binarias/numéricas
- **Variable objetivo:** `churn` (0: Cliente activo, 1: Cliente canceló)
- **Distribución de clases:**
  - No churn (0): 6,367 (63.67%)
  - Churn (1): 3,633 (36.33%)

### Variables Principales
- `tenure_months`: Meses de antigüedad del cliente
- `monthly_charges`: Cargo mensual
- `total_charges`: Cargos totales acumulados
- Features categóricas: tipo de contrato, servicios contratados, método de pago, etc.

### Preprocesamiento Realizado
1. Normalización de nombres de columnas
2. Conversión de `total_charges` a numérico
3. Imputación de valores faltantes (mediana)
4. Eliminación de identificadores de cliente
5. One-hot encoding de variables categóricas (drop_first=True)
6. Estandarización de features numéricas (StandardScaler)

---

## 🤖 Modelos Implementados

### Modelo Principal: Random Forest
```yaml
Configuración:
  n_estimators: 300
  max_depth: null (sin límite)
  min_samples_split: 2
  min_samples_leaf: 1
  class_weight: balanced_subsample
  random_state: 42
```

### Experimentos Adicionales
1. **RandomForest Baseline** - Configuración conservadora (100 árboles, max_depth=10)
2. **RandomForest Optimized** - Configuración agresiva (500 árboles, sin límite de profundidad)
3. **RandomForest Regularized** - Con regularización (max_depth=8, min_samples_split=10)
4. **Logistic Regression L2** - Baseline con regularización Ridge
5. **Logistic Regression L1** - Con regularización Lasso

---

## 📈 Resultados

### Métricas del Modelo Principal (RandomForest)
```
Accuracy:    66.35%
Precision:   55.38%
Recall:      38.24%
F1-Score:    45.24%
ROC-AUC:     69.89%
```

**Interpretación:**
- El modelo tiene un desempeño moderado en la predicción de churn
- Alta especificidad (bajo falsos positivos)
- Recall mejorable - algunos casos de churn no son detectados
- ROC-AUC cercano a 0.70 indica capacidad discriminativa aceptable

### Comparación de Experimentos
Ver archivo completo en: `reports/experiments_comparison.csv`

| Experimento | Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Características |
|-------------|--------|----------|-----------|--------|----------|---------|-----------------|
| **Logistic L2** | Logistic Regression | **65.55%** | 51.92% | **70.56%** | 59.83% | **72.66%** | Regularización Ridge (L2), mejor ROC-AUC |
| Logistic L1 | Logistic Regression | 65.50% | 51.85% | 71.39% | 60.07% | 72.63% | Regularización Lasso (L1) |
| RF Regularized | Random Forest | 66.40% | 53.24% | 62.17% | 57.36% | 71.99% | Max_depth=8, min_samples_split=10 |
| RF Baseline | Random Forest | 65.90% | 52.51% | 64.65% | 57.95% | 72.15% | Conservador, 100 árboles, max_depth=10 |
| RF Optimized | Random Forest | 66.35% | **55.38%** | 38.24% | 45.24% | 69.89% | Agresivo, 500 árboles, sin límite profundidad |


### Comparación Visual
```
ROC-AUC (cuanto más alto, mejor):
Logistic L2     ████████████████████████████████████████ 72.66%
Logistic L1     ████████████████████████████████████████ 72.63%
RF Baseline     ███████████████████████████████████████  72.15%
RF Regularized  ███████████████████████████████████████  71.99%
RF Optimized    ██████████████████████████████████       69.89%

F1-Score (balance precision-recall):
Logistic L1     ████████████████████████████████████████ 60.07%
Logistic L2     ███████████████████████████████████████  59.83%
RF Baseline     ███████████████████████████████          57.95%
RF Regularized  ██████████████████████████████           57.36%
RF Optimized    ██████████████████                       45.24%
```
---

## 🔬 Implementación MLOps

### 1. Versionado de Datos (DVC)
- Dataset raw versionado: `data/raw/telco_churn.csv.dvc`
- Dataset procesado versionado: `data/processed/telco_churn_processed.csv.dvc`
- Modelo versionado: `models/model.joblib.dvc`
- **Ventaja:** Trazabilidad completa de cambios en datos y modelos

### 2. Tracking de Experimentos (MLflow)
- **Experimento:** `proyecto_telco`
- **Tracking remoto:** DagsHub
- **Métricas registradas:** accuracy, precision, recall, f1, roc_auc
- **Parámetros registrados:** tipo de modelo, hiperparámetros, config de split
- **Artifacts:** Modelo .joblib, métricas .json

### 3. Reproducibilidad
- **params.yaml:** Configuración centralizada
- **requirements.txt:** Dependencias fijadas
- **Scripts automatizados:** Entrenamiento y experimentación reproducibles
- **DVC pipelines:** Pipeline de preprocesamiento y entrenamiento

### 4. Colaboración (DagsHub)
- **Repositorio:** https://dagshub.com/Nacho/Proyecto_Telco
- **Características:**
  - Código versionado (Git)
  - Datos versionados (DVC)
  - Experimentos visualizables (MLflow)
  - Colaboración habilitada

---

## 🚀 Ejecución del Proyecto

### Instalación
```bash
# Clonar repositorio
git clone https://dagshub.com/Nacho/Proyecto_Telco.git
cd Proyecto_Laboratorio_Mineria

# Crear entorno virtual
conda create -n proyecto_mineria python=3.9 -y
conda activate proyecto_mineria

# Instalar dependencias
pip install -r requirements.txt

# Descargar datos (DVC)
dvc pull
```

### Entrenar Modelo
```bash
# Entrenamiento con configuración por defecto
python src/train.py --params params.yaml

# Ver experimentos en MLflow
mlflow ui --port 5000
# Abrir: http://localhost:5000
```

### Ejecutar Todos los Experimentos
```bash
# Ejecutar los 5 experimentos configurados
python scripts/run_experiments.py

# Registrar mejor modelo en Model Registry
python scripts/register_best_model.py \
  --experiment proyecto_telco \
  --metric roc_auc \
  --model-name TelcoChurn_Model
```

---

## 📊 Análisis y Conclusiones

### Hallazgos Principales
1. **Desbalance de clases:** El dataset tiene 36% de churn, lo cual requiere técnicas de balanceo
2. **Features importantes:** `tenure_months`, `monthly_charges` y `total_charges` son predictores clave
3. **Trade-off precision-recall:** El modelo prioriza precisión sobre recall
4. **Mejoras posibles:** 
   - Balanceo de clases (SMOTE, undersampling)
   - Feature engineering adicional
   - Ensambles de modelos

### Ventajas del Enfoque MLOps
- ✅ **Reproducibilidad total:** Cualquier experimento puede replicarse
- ✅ **Trazabilidad:** Histórico completo de cambios en datos y modelos
- ✅ **Colaboración:** Fácil compartir resultados con el equipo
- ✅ **Experimentación rápida:** Framework para probar múltiples configuraciones
- ✅ **Versionado inteligente:** Solo se almacenan diferencias, no archivos completos

### Lecciones Aprendidas
1. La importancia del versionado de datos, no solo de código
2. MLflow simplifica el tracking de experimentos masivamente
3. DVC permite trabajar con archivos grandes sin saturar Git
4. DagsHub integra todas las herramientas en una plataforma


## 📚 Referencias

1. MLflow Documentation. (2024). https://mlflow.org/docs/latest/index.html
2. DVC Documentation. (2024). https://dvc.org/doc
3. Scikit-learn User Guide. (2024). https://scikit-learn.org/stable/user_guide.html
4. DagsHub Documentation. (2024). https://dagshub.com/docs
5. Telco Customer Churn Dataset.

---

## 📝 Notas de Entrega

### Entregables Incluidos
- ✅ Código fuente completo (`src/`, `scripts/`)
- ✅ Configuraciones de experimentos (`params_experiments/`)
- ✅ Dataset procesado y versionado (DVC)
- ✅ Modelos entrenados y versionados (DVC)
- ✅ Reporte de experimentos (`reports/experiments_comparison.csv`)
- ✅ Tracking de experimentos en DagsHub/MLflow
- ✅ Documentación completa (este README)

### Acceso al Proyecto
- **Repositorio:** https://dagshub.com/Nacho/Proyecto_Telco
- **Experimentos MLflow:** https://dagshub.com/Nacho/Proyecto_Telco.mlflow
- **Datos DVC:** https://dagshub.com/Nacho/Proyecto_Telco.dvc

### Instrucciones para Evaluación
1. Clonar repositorio desde DagsHub
2. Ejecutar `dvc pull` para descargar datos
3. Revisar experimentos en pestaña "Experiments" de DagsHub
4. Ejecutar entrenamiento con `python src/train.py --params params.yaml`
5. Ver comparación de experimentos en `reports/experiments_comparison.csv`

---

## 📧 Contacto

Para consultas sobre este proyecto:
- **DagsHub:** https://dagshub.com/Nacho

---

**Proyecto desarrollado como parte del Trabajo Práctico Integrador de Laboratorio de Minería de Datos**

*Octubre 2025*