# 📊 Informe Final - TelcoVision

**Trabajo Práctico Integrador - Laboratorio de Minería de Datos**

---

## 📋 Información del Proyecto

| Campo | Detalle |
|-------|---------|
| **Nombre del Proyecto** | TelcoVision - Sistema MLOps para Predicción de Churn |
| **Estudiantes** | Evelyn Solange Irusta, Ignacio Heck |
| **Institución** | ISTEA |
| **Materia** | Laboratorio de Minería de Datos |
| **Fecha de Entrega** | Noviembre 2025 |
| **Repositorio GitHub** | https://github.com/Shaftyel/Proyecto_telco |
| **Repositorio DagsHub** | https://dagshub.com/Nacho/proyecto_telco |

---

## 🎯 Resumen Ejecutivo

TelcoVision es un sistema completo de Machine Learning implementado con prácticas MLOps profesionales para predecir la cancelación de servicios (churn) en clientes de telecomunicaciones. El proyecto abarca desde la ingesta y preprocesamiento de datos hasta el deployment en producción, incluyendo versionado de datos/modelos, tracking de experimentos, CI/CD automatizado y múltiples opciones de despliegue.

### Resultados Clave

- ✅ **Pipeline automatizado** de 3 stages con DVC
- ✅ **6 experimentos** ejecutados con metodología colaborativa
- ✅ **Modelo ganador:** Random Forest Conservador (ROC-AUC: 72.53%, Recall: 64.65%)
- ✅ **CI/CD automatizado** validando cada cambio
- ✅ **4 opciones de deployment** documentadas y listas para producción
- ✅ **Visualizaciones profesionales** (matriz de confusión, curva ROC, curva PR)

---

## 📊 1. Problema de Negocio

### Contexto

En la industria de telecomunicaciones, la retención de clientes es crítica para la rentabilidad. Adquirir nuevos clientes es entre 5-25 veces más costoso que retener los existentes. La predicción temprana de churn permite implementar estrategias de retención proactivas.

### Objetivo

Desarrollar un sistema de predicción de churn que:
1. Identifique clientes en riesgo con **alta precisión** (recall > 60%)
2. Sea **reproducible y escalable** mediante MLOps
3. Permita **experimentación rápida** de diferentes modelos
4. Esté **listo para producción** con opciones de deployment

### Métricas de Éxito

- **Recall > 60%:** Detectar al menos 60% de casos de churn
- **ROC-AUC > 70%:** Capacidad discriminativa sólida
- **Pipeline reproducible:** Cualquier experimento replicable
- **CI/CD funcional:** Validación automática de cambios

---

## 📈 2. Datos y Preprocesamiento

### Dataset

- **Fuente:** Telco Customer Churn Dataset
- **Registros:** 10,000 clientes
- **Features originales:** 12 (9 categóricas, 3 numéricas)
- **Variable objetivo:** `churn` (binaria: 0 = activo, 1 = canceló)
- **Distribución de clases:**
  - No churn: 6,367 (63.67%)
  - Churn: 3,633 (36.33%)

### Pipeline de Preprocesamiento

**Stage: `data_prep`**

```python
# Transformaciones aplicadas:
1. Normalización de nombres de columnas
2. Conversión de tipos de datos
3. Imputación de valores faltantes (mediana)
4. Eliminación de identificadores
5. One-hot encoding (drop_first=True)
6. Estandarización (StandardScaler)
```

**Resultado:** 24 features procesadas

**Versionado:** `data/processed/telco_churn_processed.csv.dvc`

---

## 🤖 3. Experimentación y Selección de Modelo

### Metodología

Se implementó un **workflow colaborativo** con:
- Git branches por experimento (`feat-experimento-*`)
- Pull Requests para revisión
- Validación automática con CI/CD
- Tracking en MLflow/DagsHub

### Experimentos Ejecutados

| # | Configuración | Autor | Accuracy | Recall | ROC-AUC | Status |
|---|--------------|-------|----------|--------|---------|--------|
| 1 | RF 500 árboles | Nacho | 66.7% | 51.0% | 71.0% | ❌ Cerrado |
| 2 | RF Regularizado | Nacho | 66.6% | 62.6% | 72.0% | ❌ Cerrado (Subcampeón) |
| 3 | RF Balanceado | Nacho | 67.2% | 56.8% | 71.6% | ❌ Cerrado |
| 4 | RF Alto Rendimiento | Solange | 67.1% | 42.9% | 70.2% | ❌ Cerrado |
| **5** | **RF Conservador** | **Solange** | **66.7%** | **64.7%** 🥇 | **72.5%** 🥇 | ✅ **MERGED** |
| 6 | RF Equilibrado | Solange | 67.5% | 56.5% | 71.8% | ❌ Cerrado |

### Configuración del Modelo Ganador

**Experimento 5: Random Forest Conservador**

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

### Justificación de Selección

El **Experimento 5** fue seleccionado como modelo ganador por:

1. **🥇 Mejor ROC-AUC (72.53%)**
   - Superior capacidad de discriminación entre clases
   - Indica que el modelo ordena correctamente las probabilidades

2. **🥇 Mejor Recall (64.65%)**
   - Detecta 646 de cada 1000 clientes que harán churn
   - Crítico para el negocio: minimiza casos perdidos

3. **🥇 Mejor F1-Score (58.49%)**
   - Mejor balance entre precisión y recall
   - Demuestra consistencia del modelo

4. **💼 Impacto de Negocio**
   - En telecomunicaciones, detectar clientes en riesgo > accuracy general
   - Un falso positivo (ofrecer retención innecesaria) tiene bajo costo
   - Un falso negativo (perder cliente) tiene alto costo

### Comparación con Subcampeón

| Métrica | Exp 5 (Ganador) | Exp 2 (Subcampeón) | Diferencia |
|---------|-----------------|--------------------|-----------| 
| ROC-AUC | 72.53% | 72.00% | +0.53% |
| Recall | 64.65% | 62.59% | +2.06% |
| F1-Score | 58.49% | 57.67% | +0.82% |

El Experimento 5 supera al subcampeón en todas las métricas críticas para el negocio.

---

## 📊 4. Resultados del Modelo Final

### Métricas en Datos de Prueba (2000 muestras)

```
Accuracy:    66.65%
Precision:   53.41%
Recall:      64.65% ⭐
F1-Score:    58.49%
ROC-AUC:     72.53% ⭐
```

### Matriz de Confusión

```
                    Predicho
                 No Churn | Churn
         ─────────────────────────
Real     No Churn |  863  |  410
         Churn    |  257  |  470
```

**Interpretación:**
- **True Positives (470):** Casos de churn correctamente identificados
- **False Negatives (257):** Casos de churn perdidos (35.35% de todos los churn)
- **False Positives (410):** Clientes sin riesgo marcados como churn
- **True Negatives (863):** Clientes activos correctamente identificados

### Curva ROC

**AUC: 0.7253**

La curva ROC muestra que el modelo tiene capacidad discriminativa sólida, muy superior a un clasificador aleatorio (AUC = 0.50).

### Curva Precision-Recall

**Average Precision: 0.5773**

Dado el desbalance de clases, esta métrica es más informativa que accuracy. El modelo mantiene precisión razonable incluso con recall alto.

### Reporte de Clasificación

```
              precision    recall  f1-score   support

    No Churn       0.77      0.68      0.72      1273
       Churn       0.53      0.65      0.58       727

    accuracy                           0.67      2000
   macro avg       0.65      0.66      0.65      2000
weighted avg       0.68      0.67      0.67      2000
```

---

## 🔄 5. Pipeline MLOps Implementado

### Arquitectura

```
┌─────────────┐
│  Raw Data   │
└──────┬──────┘
       │ DVC tracked
       ▼
┌─────────────┐
│ data_prep   │ ← src/data_prep.py
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Processed   │
│   Data      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   train     │ ← src/train.py + MLflow
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Model +   │
│  Metrics    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  evaluate   │ ← src/evaluate.py
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Plots +   │
│  Reports    │
└─────────────┘
```

### Componentes del Sistema

#### 1. Versionado de Datos (DVC)

```bash
# Archivos versionados:
data/raw/telco_churn.csv.dvc
data/processed/telco_churn_processed.csv.dvc
models/model.joblib.dvc
```

**Beneficios:**
- Trazabilidad completa de cambios en datos
- Colaboración sin duplicar archivos grandes
- Rollback a versiones anteriores

#### 2. Tracking de Experimentos (MLflow)

```python
# Cada experimento registra:
- Parámetros: n_estimators, max_depth, etc.
- Métricas: accuracy, precision, recall, f1, roc_auc
- Artifacts: modelo .joblib, métricas .json
- Tags: autor, versión, propósito
```

**Repositorio:** https://dagshub.com/Nacho/proyecto_telco.mlflow

#### 3. CI/CD (GitHub Actions)

**Workflow:** `.github/workflows/ci.yaml`

**Triggers:**
- Push a `main`, `dev`, `feat-*`
- Pull Requests a `main`

**Jobs ejecutados:**
1. Setup Python 3.9
2. Instalar dependencias
3. Ejecutar pipeline DVC completo
4. Validar métricas (accuracy > 60%)
5. Subir artefactos
6. Tracking a MLflow

**Resultado:** Validación automática de cada experimento

#### 4. Colaboración (GitHub + DagsHub)

**GitHub:** https://github.com/Shaftyel/Proyecto_telco
- Control de versiones del código
- Pull Requests y code review
- Issues y project management

**DagsHub:** https://dagshub.com/Nacho/proyecto_telco
- Experimentos MLflow
- Storage DVC
- Visualización de métricas

---

## 🚀 6. Estrategia de Despliegue en Producción

Se documentaron **4 opciones de deployment** en `GUIA_DESPLIEGUE.md`:

### Opción 1: API REST (FastAPI) ⚡

**Características:**
- API REST production-ready
- Documentación automática (Swagger)
- Endpoints: `/predict`, `/predict_batch`, `/health`
- Validación con Pydantic
- Performance: <100ms por predicción

**Casos de uso:**
- Integración con sistemas existentes
- Aplicaciones móviles/web
- Microservicios

**Deployment:**
- Servidor: systemd service
- Cloud: AWS Elastic Beanstalk, GCP Cloud Run, Azure App Service
- Containerizado: Docker + Kubernetes

### Opción 2: Dashboard Web (Streamlit) 🎨

**Características:**
- Interface web interactiva
- Visualizaciones con Plotly
- Gauge de probabilidad de churn
- Recomendaciones automáticas

**Casos de uso:**
- Demos para stakeholders
- Herramienta interna para analistas
- Prototipado rápido

**Deployment:**
- Streamlit Cloud (gratis)
- Servidor interno

### Opción 3: Procesamiento por Lotes 📦

**Características:**
- Script para múltiples clientes
- Generación de reportes CSV
- Ejecución programada (cron)

**Casos de uso:**
- Scoring mensual de toda la base
- Reportes periódicos
- ETL pipelines

### Opción 4: Containerización (Docker) 🐳

**Características:**
- Dockerfile completo
- docker-compose con múltiples servicios
- Portabilidad total

**Casos de uso:**
- Deployment agnóstico de plataforma
- Entornos de desarrollo idénticos
- Escalamiento horizontal

### Recomendación para Producción

**Arquitectura sugerida:**

```
┌─────────────────────────────────────────┐
│         Load Balancer (Nginx)           │
└───────────┬─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│    FastAPI Container (3 instancias)     │
│    - Predicción en tiempo real          │
│    - Health checks                       │
└───────────┬─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│         PostgreSQL Database             │
│    - Logging de predicciones            │
│    - Audit trail                         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│      Batch Job (Scheduled)              │
│    - Scoring mensual completo           │
│    - Re-entrenamiento automático        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│    Monitoring (Prometheus + Grafana)    │
│    - Métricas de performance            │
│    - Data drift detection               │
└─────────────────────────────────────────┘
```

### Monitoreo Post-Deployment

**Métricas a trackear:**
1. **Performance del modelo:**
   - Accuracy, Precision, Recall en producción
   - Data drift (cambios en distribución de datos)
   - Prediction drift (cambios en distribución de predicciones)

2. **Performance del sistema:**
   - Latencia de respuesta (target: <100ms)
   - Throughput (requests/segundo)
   - Uso de recursos (CPU, memoria)
   - Tasa de errores

3. **Métricas de negocio:**
   - Tasa de churn real vs predicho
   - ROI de acciones preventivas
   - Costo de falsos positivos/negativos

### Re-entrenamiento

**Estrategia:**
- **Frecuencia:** Mensual
- **Trigger:** Detección de data drift OR performance degradation
- **Pipeline:** Automatizado con DVC + GitHub Actions
- **Validación:** A/B testing antes de reemplazar modelo actual

---

## 💡 7. Aprendizajes y Mejores Prácticas

### Técnicos

#### Lo que funcionó bien ✅

1. **DVC para versionado de datos**
   - Simplicidad: archivos `.dvc` pequeños en Git
   - Eficiencia: solo se almacenan diferencias
   - Colaboración: múltiples personas sin conflictos

2. **MLflow para tracking**
   - Comparación visual de experimentos
   - Historial completo de runs
   - Artifacts versionados automáticamente

3. **GitHub Actions para CI/CD**
   - Validación automática de cada cambio
   - Prevención de errores en main
   - Feedback rápido (< 5 minutos)

4. **Branches + PRs para experimentación**
   - Experimentar sin romper main
   - Code review estructurado
   - Historial claro de decisiones

#### Desafíos enfrentados ⚠️

1. **Integración DVC + GitHub**
   - Problema: Archivos grandes en Git
   - Solución: Asegurar que `.dvc` files están en .gitignore

2. **Encoding en Windows**
   - Problema: Emojis en scripts causaban errores
   - Solución: Usar solo ASCII en código crítico

3. **Feature importance en modelo**
   - Problema: RF no exponía feature_importances_
   - Solución: Generar placeholder o quitar del pipeline

### MLOps

#### Principios aplicados 📐

1. **Everything as Code**
   - Configuración: `params.yaml`
   - Pipeline: `dvc.yaml`
   - CI/CD: `.github/workflows/ci.yaml`

2. **Reproducibilidad**
   - Random seeds fijos
   - Dependencias versionadas
   - Entornos aislados (conda)

3. **Automatización**
   - Pipeline completo con `dvc repro`
   - CI/CD sin intervención manual
   - Tracking automático de experimentos

4. **Colaboración**
   - Git workflow estándar
   - Pull Requests para cambios
   - Documentación actualizada

#### Lecciones aprendidas 🎓

1. **Versionado de datos es tan importante como código**
   - Los modelos dependen de los datos
   - Reproducibilidad requiere versionar ambos

2. **Documentación temprana ahorra tiempo**
   - README claro facilita onboarding
   - Guías reducen preguntas repetitivas

3. **CI/CD previene errores costosos**
   - Validación automática > revisión manual
   - Feedback rápido > debugging tardío

4. **Métricas deben alinearse con negocio**
   - Accuracy != métrica correcta siempre
   - En churn, recall > precision

### Mejoras Futuras

#### Corto Plazo (1-2 semanas)
- [ ] Implementar SMOTE para balanceo de clases
- [ ] Probar XGBoost y LightGBM
- [ ] Agregar feature engineering (ratios, interacciones)
- [ ] Implementar API REST en staging

#### Mediano Plazo (1-2 meses)
- [ ] Hyperparameter tuning con Optuna
- [ ] Ensamble de modelos (voting, stacking)
- [ ] Dashboard de monitoreo con Grafana
- [ ] A/B testing en producción

#### Largo Plazo (3+ meses)
- [ ] Incorporar datos de tiempo real (streaming)
- [ ] Explainability con SHAP values
- [ ] Predicción de Customer Lifetime Value
- [ ] Recomendaciones personalizadas de retención

---

## 📊 8. Impacto y Resultados

### Métricas del Proyecto

| Aspecto | Métrica | Resultado |
|---------|---------|-----------|
| **Experimentación** | Número de experimentos | 6 |
| **Colaboración** | Número de colaboradores | 2 (Nacho + Solange) |
| **Automatización** | Pipeline stages | 3 (data_prep, train, evaluate) |
| **CI/CD** | Pull Requests validados | 6 |
| **Documentación** | Páginas de docs | 5 archivos .md |
| **Deployment** | Opciones documentadas | 4 (API, Streamlit, Batch, Docker) |

### Impacto de Negocio Proyectado

**Escenario:** Empresa de telecomunicaciones con 100,000 clientes

**Métricas actuales (sin modelo):**
- Tasa de churn mensual: 3% (3,000 clientes/mes)
- Costo de adquisición por cliente: $500
- Pérdida mensual: $1,500,000

**Con modelo implementado:**

Usando Recall = 64.65%:
- Clientes en riesgo detectados: 1,940 de 3,000
- Costo de retención por cliente: $100
- Inversión en retención: $194,000

Si logramos retener 40% de los contactados:
- Clientes retenidos: 776
- Ahorro en costo de re-adquisición: $388,000
- **ROI mensual: $194,000** (100% en primer mes)

**ROI anual proyectado: $2,328,000**

---

## ✅ 9. Cumplimiento de Objetivos

### Objetivos Planteados vs Alcanzados

| Objetivo | Meta | Alcanzado | Status |
|----------|------|-----------|--------|
| Pipeline reproducible | DVC funcional | 3 stages automatizados | ✅ |
| Tracking de experimentos | MLflow integrado | 6 experimentos tracked | ✅ |
| CI/CD automatizado | GitHub Actions | Validación en cada PR | ✅ |
| Modelo con recall > 60% | 60%+ | 64.65% | ✅ |
| Modelo con ROC-AUC > 70% | 70%+ | 72.53% | ✅ |
| Documentación completa | README + guías | 5 archivos .md | ✅ |
| Opciones de deployment | Al menos 2 | 4 documentadas | ✅ |
| Colaboración remota | DagsHub funcional | GitHub + DagsHub | ✅ |

**Resultado: 8/8 objetivos cumplidos** 🎉

---

## 📚 10. Referencias

### Documentación del Proyecto
- **README.md** - Visión general completa
- **GUIA_COLABORADOR.md** - Onboarding para nuevos miembros
- **GUIA_DESPLIEGUE.md** - Opciones de deployment
- **COMPARACION_EXPERIMENTOS_FINAL.md** - Análisis detallado de experimentos

### Repositorios
- **GitHub:** https://github.com/Shaftyel/Proyecto_telco
- **DagsHub:** https://dagshub.com/Nacho/proyecto_telco
- **MLflow:** https://dagshub.com/Nacho/proyecto_telco.mlflow

### Tecnologías
- [MLflow Documentation](https://mlflow.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/)

### Papers y Recursos
- MLOps: Continuous delivery and automation pipelines in ML (Google Cloud)
- Reproducibility in Machine Learning (Papers with Code)

---

## 👥 11. Equipo y Contribuciones

### Nacho
**Rol:** Project Owner / ML Engineer

**Contribuciones:**
- Setup inicial del proyecto (Git, DVC, MLflow)
- Pipeline de preprocesamiento (`data_prep.py`)
- Script de entrenamiento con tracking (`train.py`)
- Experimentos 1, 2, 3 (500 árboles, regularizado, balanceado)
- Configuración CI/CD (GitHub Actions)
- Integración con DagsHub

### Solange
**Rol:** Colaboradora / Data Scientist

**Contribuciones:**
- Experimentos 4, 5, 6 (alto rendimiento, conservador, equilibrado)
- Script de evaluación avanzada (`evaluate.py`)
- Identificación del modelo ganador
- Visualizaciones (matriz confusión, curvas ROC/PR)

### Trabajo Conjunto
- Documentación completa (README, guías)
- Análisis comparativo de experimentos
- Estrategia de deployment
- Preparación de entregables

---

## 🎯 12. Conclusiones

### Logros Principales

1. **Sistema MLOps Completo**
   - Pipeline reproducible de principio a fin
   - Versionado de datos, código y modelos
   - CI/CD validando cada cambio

2. **Modelo Production-Ready**
   - Recall: 64.65% (detecta 2 de cada 3 casos de churn)
   - ROC-AUC: 72.53% (capacidad discriminativa sólida)
   - 4 opciones de deployment documentadas

3. **Experimentación Estructurada**
   - 6 experimentos con metodología consistente
   - Tracking completo en MLflow
   - Selección justificada del mejor modelo

4. **Documentación Profesional**
   - 5 archivos markdown completos
   - Guías para colaboradores y deployment
   - README exhaustivo

### Valor Agregado del Proyecto

**Técnico:**
- Framework replicable para futuros proyectos ML
- Best practices de MLOps documentadas
- Código production-ready

**Académico:**
- Aplicación práctica de conceptos de minería de datos
- Integración de múltiples herramientas (DVC, MLflow, GitHub Actions)
- Experiencia en trabajo colaborativo

**Profesional:**
- Portfolio demostrable de habilidades MLOps
- Experiencia en ciclo completo de proyecto ML
- Documentación nivel empresarial

### Reflexión Final

Este proyecto demuestra que implementar MLOps no es solo "una buena práctica" sino una **necesidad** para proyectos de ML serios. La inversión en versionado, tracking y automatización se paga rápidamente en:

1. **Tiempo ahorrado** en debugging y reproducción
2. **Calidad mejorada** con validación automática
3. **Colaboración facilitada** con workflow estructurado
4. **Confianza incrementada** en resultados reproducibles

El modelo final, con recall de 64.65%, representa un balance práctico entre detectar churn y evitar falsos positivos. En un contexto real de negocio, este modelo podría generar ROI significativo al permitir acciones proactivas de retención.

---

## 📎 Anexos

### A. Estructura Completa del Repositorio

Ver: `README.md` sección "Estructura del Proyecto"

### B. Comandos Útiles

```bash
# Setup inicial
git clone https://github.com/Shaftyel/Proyecto_telco.git
conda create -n telcovision python=3.9 -y
conda activate telcovision
pip install -r requirements.txt

# Ejecutar pipeline
dvc repro

# Ver experimentos
mlflow ui

# Ejecutar API
uvicorn src.api:app --reload

# Ejecutar dashboard
streamlit run src/app_streamlit.py
```

### C. Métricas Detalladas de Todos los Experimentos

Ver: `COMPARACION_EXPERIMENTOS_FINAL.md`

### D. Guía de Deployment Paso a Paso

Ver: `GUIA_DESPLIEGUE.md`

---

<div align="center">

## 🎓 Trabajo Práctico Integrador Completado

**TelcoVision - Sistema MLOps para Predicción de Churn**

Evelyn Solange Irusta • Ignacio Heck

ISTEA - Laboratorio de Minería de Datos

Noviembre 2025

---

*"En MLOps, la reproducibilidad no es opcional - es fundamental"*

</div>
