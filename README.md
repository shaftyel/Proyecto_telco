# 📊 TelcoVision - Sistema MLOps para Predicción de Churn

**Trabajo Práctico Integrador - Laboratorio de Minería de Datos**

Sistema completo de Machine Learning con versionado de datos, tracking de experimentos, CI/CD automatizado y múltiples opciones de despliegue para predecir la cancelación de servicios (churn) en clientes de telecomunicaciones.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.15.0-blue.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.50.0-purple.svg)](https://dvc.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Ready-green.svg)](https://fastapi.tiangolo.com)

---

## 👥 Equipo

- **Estudiantes:** 
  - Evelyn Solange Irusta
  - Ignacio Heck
- **Institución:** ISTEA
- **Materia:** Laboratorio de Minería de Datos
- **Fecha:** Noviembre 2025

---

## 🎯 Objetivo

Implementar un pipeline end-to-end de Machine Learning con prácticas MLOps profesionales:
- ✅ Versionado de datos y modelos (DVC)
- ✅ Tracking de experimentos (MLflow)
- ✅ CI/CD automatizado (GitHub Actions)
- ✅ Colaboración remota (DagsHub + GitHub)
- ✅ Evaluación avanzada con visualizaciones
- ✅ Múltiples opciones de despliegue en producción
- ✅ Reproducibilidad total del proyecto

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    TELCOVISION ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

DATA LAYER                 PROCESSING LAYER           DEPLOYMENT
┌──────────┐              ┌──────────────┐           ┌──────────┐
│ Raw Data │──DVC────────▶│ data_prep.py │           │ FastAPI  │
│ 10k rows │              └──────┬───────┘           │   API    │
└──────────┘                     │                   └────┬─────┘
                                 ▼                        │
                          ┌──────────────┐               │
                          │  Processed   │               │
                          │    Data      │               │
                          └──────┬───────┘               │
                                 │                        │
EXPERIMENT TRACKING              ▼                        ▼
┌──────────┐              ┌──────────────┐           ┌──────────┐
│  MLflow  │◀────────────│   train.py   │           │Streamlit │
│  Runs    │              │ 6 exps       │           │Dashboard │
└──────────┘              └──────┬───────┘           └──────────┘
                                 │
                                 ▼                    ┌──────────┐
VERSION CONTROL           ┌──────────────┐           │  Docker  │
┌──────────┐              │  evaluate.py │           │Container │
│   Git    │◀────────────│ Visualizations│           └──────────┘
│   DVC    │              └──────┬───────┘
└──────────┘                     │
                                 ▼
CI/CD                      ┌──────────────┐
┌──────────┐              │   Model +     │
│  GitHub  │              │  Artifacts    │
│  Actions │──────────────│  (versioned)  │
└──────────┘              └───────────────┘
```

---

## 📁 Estructura del Proyecto

```
Proyecto_Laboratorio_MineriaV2/
├── .github/
│   └── workflows/
│       └── ci.yaml                     # CI/CD automatizado con GitHub Actions
├── data/
│   ├── raw/
│   │   └── telco_churn.csv            # Dataset original (10,000 registros)
│   └── processed/
│       └── telco_churn_processed.csv  # Dataset limpio y transformado
├── models/
│   ├── model.joblib                   # Modelo ganador (RF Conservador)
│   └── metrics.json                   # Métricas principales
├── metrics/
│   ├── classification_report.json     # Reporte de clasificación detallado
│   └── evaluation_summary.json        # Resumen ejecutivo de evaluación
├── plots/
│   ├── confusion_matrix.png           # Matriz de confusión
│   ├── roc_curve.png                  # Curva ROC (AUC: 0.7253)
│   └── precision_recall_curve.png     # Curva Precision-Recall
├── src/
│   ├── data_prep.py                   # Preprocesamiento de datos
│   ├── train.py                       # Entrenamiento con MLflow tracking
│   └── evaluate.py                    # Evaluación avanzada con visualizaciones
├── .dvc/                              # Configuración DVC
├── mlruns/                            # Experimentos MLflow (local)
├── params.yaml                        # Configuración del modelo ganador
├── dvc.yaml                           # Pipeline DVC (3 stages)
├── requirements.txt                   # Dependencias del proyecto
├── GUIA_COLABORADOR.md               # Guía completa para nuevos colaboradores
├── GUIA_DESPLIEGUE.md                # Documentación de deployment (FastAPI, Streamlit, Docker)
├── COMPARACION_EXPERIMENTOS_FINAL.md # Análisis detallado de 6 experimentos
└── README.md                          # Este archivo
```

---

## 🔧 Tecnologías Utilizadas

### MLOps Stack
| Herramienta | Propósito | Versión |
|-------------|-----------|---------|
| **Python** | Lenguaje principal | 3.9+ |
| **DVC** | Versionado de datos/modelos | 3.50.0 |
| **MLflow** | Tracking de experimentos | 2.15.0 |
| **GitHub Actions** | CI/CD automatizado | - |
| **DagsHub** | Colaboración MLOps | - |

### Machine Learning
| Herramienta | Propósito | Versión |
|-------------|-----------|---------|
| **scikit-learn** | Modelos ML | 1.3.0 |
| **pandas** | Manipulación de datos | 2.0.3 |
| **numpy** | Computación numérica | 1.24.0 |

### Visualización y Evaluación
| Herramienta | Propósito | Versión |
|-------------|-----------|---------|
| **matplotlib** | Gráficos estáticos | 3.7.0 |
| **seaborn** | Visualizaciones estadísticas | 0.12.0 |
| **plotly** | Gráficos interactivos | 5.17.0 |

### Deployment (Opcional)
| Herramienta | Propósito | Versión |
|-------------|-----------|---------|
| **FastAPI** | API REST | 0.104.0+ |
| **Streamlit** | Dashboard web | 1.28.0+ |
| **Docker** | Containerización | 20.10+ |
| **uvicorn** | ASGI server | 0.24.0+ |

---

## 📊 Dataset

### Características
- **Nombre:** Telco Customer Churn Dataset
- **Registros:** 10,000 clientes
- **Features procesadas:** 24 (después de encoding)
  - **Originales:** 12 features (9 categóricas, 3 numéricas)
  - **Transformadas:** 24 features binarias/numéricas
- **Variable objetivo:** `churn` (0: Cliente activo, 1: Cliente canceló)
- **Distribución de clases:**
  - No churn (0): 6,367 (63.67%)
  - Churn (1): 3,633 (36.33%)

### Variables Principales
- `tenure_months`: Meses de antigüedad del cliente
- `monthly_charges`: Cargo mensual
- `total_charges`: Cargos totales acumulados
- Features categóricas: tipo de contrato, servicios contratados, método de pago, etc.

### Pipeline de Preprocesamiento
1. ✅ Normalización de nombres de columnas
2. ✅ Conversión de `total_charges` a numérico
3. ✅ Imputación de valores faltantes (mediana)
4. ✅ Eliminación de identificadores de cliente
5. ✅ One-hot encoding (drop_first=True)
6. ✅ Estandarización de features numéricas (StandardScaler)

**Script:** `src/data_prep.py`

---

## 🧪 Experimentación y Selección de Modelo

### Metodología de Experimentación

Se realizaron **6 experimentos** con diferentes configuraciones de Random Forest, utilizando un workflow colaborativo con Git branches y Pull Requests. Cada experimento fue:
- ✅ Ejecutado en rama separada (`feat-*`)
- ✅ Validado automáticamente por CI/CD
- ✅ Trackeado en MLflow/DagsHub
- ✅ Revisado mediante Pull Request

### Comparación de Experimentos

| # | Configuración | Accuracy | Precision | Recall | F1 | ROC-AUC | Autor |
|---|--------------|----------|-----------|--------|-----|---------|-------|
| 1 | RF 500 árboles | 66.7% | 54.5% | 51.0% | 52.7% | 71.0% | Nacho |
| 2 | RF Regularizado | 66.6% | 53.5% | 62.6% | 57.7% | 72.0% | Nacho |
| 3 | RF Balanceado | 67.2% | 54.6% | 56.8% | 55.7% | 71.6% | Nacho |
| 4 | RF Alto Rendimiento | 67.1% | 56.1% | 42.9% | 48.6% | 70.2% | Solange |
| **5** | **RF Conservador** ⭐ | **66.7%** | **53.4%** | **64.7%** 🥇 | **58.5%** 🥇 | **72.5%** 🥇 | **Solange** |
| 6 | RF Equilibrado | **67.5%** 🥇 | 55.2% | 56.5% | 55.8% | 71.8% | Solange |

### 🏆 Modelo Ganador: Experimento 5 - Random Forest Conservador

**Justificación:**
- 🥇 **Mejor ROC-AUC (72.53%):** Superior capacidad de discriminación
- 🥇 **Mejor Recall (64.65%):** Detecta el 65% de los casos de churn
- 🥇 **Mejor F1-Score (58.49%):** Mejor balance precision-recall
- 💼 **Impacto de negocio:** En telecomunicaciones, detectar clientes en riesgo es más valioso que accuracy general

**Configuración del modelo ganador:**
```yaml
model:
  type: RandomForest
  parameters:
    n_estimators: 180
    max_depth: 14
    min_samples_split: 12
    min_samples_leaf: 6
    class_weight: balanced_subsample
    random_state: 42
```

**Ver análisis completo:** `COMPARACION_EXPERIMENTOS_FINAL.md`

---

## 📈 Resultados del Modelo en Producción

### Métricas Principales
```
Accuracy:    66.65%
Precision:   53.41%
Recall:      64.65% ⭐ (detecta 65% de casos de churn)
F1-Score:    58.49%
ROC-AUC:     72.53% ⭐
```

### Matriz de Confusión

```
                 Predicho No Churn  |  Predicho Churn
─────────────────────────────────────────────────────
Real No Churn         863          |       410
Real Churn            257          |       470
```

**Interpretación:**
- **True Positives (470):** Clientes correctamente identificados como churn
- **False Negatives (257):** Casos de churn no detectados (35.35%)
- **False Positives (410):** Clientes sin churn marcados como riesgo

### Visualizaciones Generadas

El pipeline genera automáticamente:

1. **Matriz de Confusión** (`plots/confusion_matrix.png`)
   - Heatmap con valores absolutos
   - Visualización clara de predicciones correctas/incorrectas

2. **Curva ROC** (`plots/roc_curve.png`)
   - AUC: 0.7253
   - Muestra capacidad discriminativa del modelo

3. **Curva Precision-Recall** (`plots/precision_recall_curve.png`)
   - Average Precision: 0.5773
   - Balance entre precisión y recall

### Reporte de Clasificación Completo

```
              precision    recall  f1-score   support

    No Churn       0.77      0.68      0.72      1273
       Churn       0.53      0.65      0.58       727

    accuracy                           0.67      2000
   macro avg       0.65      0.66      0.65      2000
weighted avg       0.68      0.67      0.67      2000
```

**Ver detalles:** `metrics/classification_report.json`

---

## 🔄 Pipeline Automatizado

### Arquitectura del Pipeline (DVC)

```
┌─────────────────────┐
│  data/raw/          │
│  telco_churn.csv    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  [data_prep]        │
│  Preprocessing      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  data/processed/    │
│  cleaned data       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  [train]            │
│  Model training     │
│  + MLflow tracking  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  models/            │
│  model.joblib       │
│  metrics.json       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  [evaluate]         │
│  Advanced metrics   │
│  + Visualizations   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  plots/             │
│  *.png              │
│  metrics/*.json     │
└─────────────────────┘
```

### Stages del Pipeline

**1. data_prep** - Preprocesamiento
```bash
python src/data_prep.py
```
- Input: `data/raw/telco_churn.csv`
- Output: `data/processed/telco_churn_processed.csv`

**2. train** - Entrenamiento
```bash
python src/train.py --params params.yaml
```
- Input: Datos procesados + `params.yaml`
- Output: `models/model.joblib` + `models/metrics.json`
- Tracking: MLflow run con parámetros y métricas

**3. evaluate** - Evaluación Avanzada
```bash
python src/evaluate.py
```
- Input: Modelo + datos procesados
- Output: 
  - `plots/confusion_matrix.png`
  - `plots/roc_curve.png`
  - `plots/precision_recall_curve.png`
  - `metrics/classification_report.json`
  - `metrics/evaluation_summary.json`

### Ejecutar Pipeline Completo

```bash
# Ejecutar todo el pipeline
dvc repro

# Ver métricas
cat models/metrics.json
cat metrics/evaluation_summary.json

# Visualizar plots
ls plots/
```

---

## 🤝 Colaboración y CI/CD

### Workflow de Desarrollo

```
1. Crear rama experimental
   git checkout -b feat-nuevo-experimento

2. Modificar configuración
   vim params.yaml

3. Ejecutar experimento
   dvc repro

4. Commitear cambios
   git add params.yaml dvc.lock
   git commit -m "feat: nuevo experimento"

5. Push y crear PR
   git push origin feat-nuevo-experimento

6. CI/CD valida automáticamente
   ✓ Instala dependencias
   ✓ Ejecuta pipeline
   ✓ Valida métricas (accuracy > 60%)
   ✓ Guarda artefactos

7. Revisión y merge
   Code review → Merge a main
```

### GitHub Actions CI/CD

**Archivo:** `.github/workflows/ci.yaml`

**Triggers:**
- Push a `main`, `dev`, `feat-*`
- Pull requests a `main`

**Jobs:**
1. ✅ Setup Python 3.9
2. ✅ Instalar dependencias
3. ✅ Ejecutar pipeline DVC
4. ✅ Validar métricas mínimas
5. ✅ Subir artefactos
6. ✅ Tracking a MLflow/DagsHub

### Colaboración Remota

**GitHub:** https://github.com/Shaftyel/Proyecto_telco
- Control de versiones del código
- Pull Requests y code review
- CI/CD con GitHub Actions

**DagsHub:** https://dagshub.com/Nacho/proyecto_telco
- Experimentos MLflow
- Datos DVC
- Visualización de métricas
- Colaboración MLOps

---

## 🚀 Ejecución del Proyecto

### Instalación Completa

```bash
# 1. Clonar repositorio
git clone https://github.com/Shaftyel/Proyecto_telco.git
cd Proyecto_telco

# 2. Crear entorno virtual
conda create -n telcovision python=3.9 -y
conda activate telcovision

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar DVC remoto (opcional)
dvc remote add -d dagshub https://dagshub.com/Nacho/proyecto_telco.dvc
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user Nacho
dvc remote modify dagshub --local password [TOKEN]

# 5. Descargar datos (si están en remoto)
dvc pull
```

### Entrenamiento Rápido

```bash
# Ejecutar pipeline completo
dvc repro

# Ver resultados
cat models/metrics.json
cat metrics/evaluation_summary.json

# Visualizar plots
open plots/confusion_matrix.png
open plots/roc_curve.png
open plots/precision_recall_curve.png
```

### Experimentación

```bash
# 1. Crear rama experimental
git checkout -b feat-mi-experimento

# 2. Modificar parámetros
vim params.yaml

# 3. Ejecutar
dvc repro

# 4. Commitear y hacer PR
git add params.yaml dvc.lock
git commit -m "feat: nuevo experimento con [descripción]"
git push origin feat-mi-experimento
```

### Tracking con MLflow

```bash
# Ver experimentos locales
mlflow ui --port 5000

# Abrir navegador
http://localhost:5000
```

---

## 🌐 Despliegue en Producción

El proyecto incluye **múltiples opciones de deployment** documentadas en `GUIA_DESPLIEGUE.md`:

### Opción 1: API REST con FastAPI ⚡

**Características:**
- API REST production-ready
- Documentación automática (Swagger/OpenAPI)
- Validación con Pydantic
- Alto rendimiento (async/await)

**Endpoints:**
- `GET /` - Info de la API
- `GET /health` - Health check
- `POST /predict` - Predicción individual
- `POST /predict_batch` - Predicción por lotes

**Ejecutar:**
```bash
# Instalar dependencias adicionales
pip install fastapi uvicorn pydantic

# Ejecutar API
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Acceder a documentación
http://localhost:8000/docs
```

**Ejemplo de uso:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "tenure": 12,
        "monthly_charges": 70.5,
        "total_charges": 846.0
    }
)

print(response.json())
# {
#   "churn_probability": 0.65,
#   "will_churn": true,
#   "risk_level": "Alto",
#   "confidence": 0.65
# }
```

---

### Opción 2: Dashboard con Streamlit 🎨

**Características:**
- Interface web interactiva
- Visualizaciones con Plotly
- Ideal para demos y uso interno
- Deploy en Streamlit Cloud (gratis)

**Ejecutar:**
```bash
# Instalar dependencias
pip install streamlit plotly

# Ejecutar dashboard
streamlit run src/app_streamlit.py

# Acceder
http://localhost:8501
```

---

### Opción 3: Batch Processing 📦

Para procesar múltiples clientes de una vez:

```bash
python src/batch_predict.py \
  --input data/clientes_nuevos.csv \
  --output predictions/batch_20251120.csv
```

---

### Opción 4: Docker 🐳

**Containerización completa:**

```bash
# Build
docker build -t telcovision-api .

# Run
docker run -p 8000:8000 telcovision-api

# O con docker-compose (API + DB + Monitoring)
docker-compose up -d
```

**Ver documentación completa:** `GUIA_DESPLIEGUE.md`

---

## 📚 Documentación Adicional

### Guías Disponibles

1. **GUIA_COLABORADOR.md**
   - Setup completo paso a paso
   - Cómo hacer experimentos
   - Workflow de Pull Requests
   - Troubleshooting común

2. **GUIA_DESPLIEGUE.md**
   - FastAPI REST API (código completo)
   - Streamlit Dashboard
   - Batch Processing
   - Docker & docker-compose
   - Deployment en cloud (AWS, GCP, Azure)
   - Seguridad y monitoreo

3. **COMPARACION_EXPERIMENTOS_FINAL.md**
   - Análisis detallado de 6 experimentos
   - Justificación del modelo ganador
   - Aprendizajes clave
   - Timeline del proyecto

### Scripts Útiles

```bash
# Ver estructura del proyecto
tree -L 3

# Ver métricas de todos los experimentos
cat reports/experiments_comparison.csv

# Ver logs de MLflow
cat mlruns/*/meta.yaml

# Limpiar archivos generados
dvc gc
git clean -fd
```

---

## 🔬 Implementación MLOps

### Principios Aplicados

#### 1. Versionado Completo 📦
- **Código:** Git
- **Datos:** DVC (`.dvc` files)
- **Modelos:** DVC + MLflow
- **Configuración:** `params.yaml` versionado

#### 2. Reproducibilidad 🔄
- Pipeline declarativo (`dvc.yaml`)
- Dependencias fijadas (`requirements.txt`)
- Random seeds controlados
- Entornos aislados (conda)

#### 3. Tracking de Experimentos 📊
- Parámetros automáticos en MLflow
- Métricas tracked en cada run
- Artifacts versionados
- Comparación visual de experimentos

#### 4. CI/CD Automatizado 🤖
- Validación automática de PRs
- Tests de pipeline
- Validación de métricas mínimas
- Deployment automatizable

#### 5. Colaboración 👥
- Código en GitHub
- Datos en DagsHub
- Experimentos compartidos
- Pull Request workflow

### Ventajas del Enfoque MLOps

✅ **Reproducibilidad total:** Cualquier experimento puede replicarse exactamente  
✅ **Trazabilidad:** Historial completo de cambios en datos, código y modelos  
✅ **Colaboración:** Framework para trabajar en equipo eficientemente  
✅ **Experimentación rápida:** Probar configuraciones sin romper main  
✅ **Versionado inteligente:** Solo se almacenan diferencias (ahorro de espacio)  
✅ **CI/CD:** Validación automática de cada cambio  
✅ **Production-ready:** Código listo para desplegar  

---

## 📊 Análisis y Conclusiones

### Hallazgos Principales

1. **Desbalance de clases (36% churn):**
   - Requiere técnicas de balanceo
   - `class_weight` fue crucial en el modelo ganador

2. **Features más importantes:**
   - `tenure_months`: Antigüedad del cliente
   - `monthly_charges`: Cargo mensual
   - `total_charges`: Acumulado histórico

3. **Trade-off precision-recall:**
   - Modelo prioriza recall (detectar churn)
   - En negocio, es mejor tener falsos positivos que perder clientes

4. **Regularización efectiva:**
   - Modelo conservador (alta regularización) ganó
   - Prevenir overfitting fue clave

### Mejoras Futuras Posibles

#### Corto Plazo
- [ ] Implementar SMOTE para balanceo de clases
- [ ] Feature engineering adicional (ratios, interacciones)
- [ ] Prueba de XGBoost y LightGBM
- [ ] Hyperparameter tuning con Optuna

#### Mediano Plazo
- [ ] API REST en producción
- [ ] Monitoreo de data drift
- [ ] Re-entrenamiento automático mensual
- [ ] A/B testing entre modelos

#### Largo Plazo
- [ ] Incorporar datos de tiempo real
- [ ] Ensambles de modelos
- [ ] Explainability con SHAP values
- [ ] Predicción de customer lifetime value

### Lecciones Aprendidas

#### Técnicas
1. **Versionado de datos es tan importante como código**
2. **MLflow simplifica el tracking masivamente**
3. **DVC permite trabajar con archivos grandes sin saturar Git**
4. **CI/CD automatizado previene errores en producción**

#### MLOps
1. **La reproducibilidad requiere disciplina pero vale la pena**
2. **Pull Requests son excelentes para experimentación colaborativa**
3. **Documentación clara es esencial para onboarding**
4. **Separar experimentación de producción (branches) es clave**

#### Negocio
1. **Las métricas deben alinearse con objetivos de negocio**
2. **En churn, recall > accuracy**
3. **La interpretabilidad del modelo importa**
4. **El costo de falsos negativos supera el de falsos positivos**

---

## 🎓 Referencias

### Documentación
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Streamlit Documentation](https://docs.streamlit.io)

### Datasets y Papers
- Telco Customer Churn Dataset
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Plataformas
- [DagsHub](https://dagshub.com)
- [GitHub Actions](https://docs.github.com/en/actions)

---

## 📝 Checklist de Entregables

### Código y Scripts
- [x] Pipeline de preprocesamiento (`src/data_prep.py`)
- [x] Script de entrenamiento con MLflow (`src/train.py`)
- [x] Script de evaluación avanzada (`src/evaluate.py`)
- [x] Configuración del modelo (`params.yaml`)
- [x] Pipeline DVC completo (`dvc.yaml`)

### Documentación
- [x] README completo (este archivo)
- [x] Guía para colaboradores (`GUIA_COLABORADOR.md`)
- [x] Guía de despliegue (`GUIA_DESPLIEGUE.md`)
- [x] Comparación de experimentos (`COMPARACION_EXPERIMENTOS_FINAL.md`)

### Artefactos
- [x] Modelo entrenado versionado (DVC)
- [x] Datos procesados versionados (DVC)
- [x] Métricas de evaluación (JSON)
- [x] Visualizaciones (PNG)
- [x] Experimentos tracked (MLflow)

### Infraestructura MLOps
- [x] Repositorio Git configurado
- [x] DVC configurado y funcional
- [x] MLflow tracking habilitado
- [x] CI/CD con GitHub Actions
- [x] Integración con DagsHub

### Deployment
- [x] Scripts de API (FastAPI)
- [x] Dashboard (Streamlit)
- [x] Batch processing
- [x] Dockerfile

---

## 🔗 Enlaces del Proyecto

### Repositorios
- **GitHub:** https://github.com/Shaftyel/Proyecto_telco
- **DagsHub:** https://dagshub.com/Nacho/proyecto_telco

### Tracking y Visualización
- **MLflow Experiments:** https://dagshub.com/Nacho/proyecto_telco.mlflow
- **DVC Remote:** https://dagshub.com/Nacho/proyecto_telco.dvc

### CI/CD
- **GitHub Actions:** https://github.com/Shaftyel/Proyecto_telco/actions

---

## 📧 Contacto

Para consultas sobre este proyecto:
- **GitHub:** [@Shaftyel](https://github.com/Shaftyel)
- **DagsHub:** [Nacho](https://dagshub.com/Nacho)
- **DagsHub:** [Solange](https://dagshub.com/SolangeIruSant)
---

## 📄 Licencia

Este proyecto fue desarrollado como parte del Trabajo Práctico Integrador de Laboratorio de Minería de Datos en ISTEA por Evelyn Solange Irusta e
Ignacio Heck.

---

<div align="center">

**TelcoVision** - Predicción de Churn con MLOps Profesional

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org)
[![MLOps](https://img.shields.io/badge/MLOps-Enabled-green.svg)]()
[![DVC](https://img.shields.io/badge/DVC-Versioned-purple.svg)](https://dvc.org)

*Noviembre 2025 - ISTEA*

</div>
