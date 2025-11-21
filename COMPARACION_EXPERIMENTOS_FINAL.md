# 🏆 Comparación Final de Experimentos - Proyecto TelcoVision

## Resumen Ejecutivo

Se realizaron **6 experimentos** con diferentes configuraciones de Random Forest para optimizar la predicción de churn en telecomunicaciones. El **Experimento 5** fue seleccionado como modelo ganador.

---

## Tabla Comparativa Completa

| # | Experimento | Autor | Accuracy | Precision | Recall | F1 | ROC-AUC | Estado |
|---|-------------|-------|----------|-----------|--------|-----|---------|--------|
| 1 | RF 500 árboles | Nacho | 66.7% | 54.48% | 51.03% | 52.70% | 71.00% | ❌ Cerrado |
| 2 | RF Regularizado | Nacho | 66.6% | 53.47% | 62.59% | 57.67% | 72.00% | ❌ Cerrado (Subcampeón) |
| 3 | RF Balanceado | Nacho | 67.15% | 54.63% | 56.81% | 55.70% | 71.60% | ❌ Cerrado |
| 4 | RF Alto Rendimiento | Solange | 67.05% | 56.12% | 42.92% | 48.64% | 70.23% | ❌ Cerrado |
| 5 | **RF Conservador** | Solange | 66.65% | 53.41% | **64.65%** | **58.49%** | **72.53%** | ✅ **MERGED** |
| 6 | RF Equilibrado | Solange | **67.5%** | 55.17% | 56.53% | 55.84% | 71.77% | ❌ Cerrado |

---

## 🏆 Experimento Ganador: #5 - RF Conservador

### Configuración:

```
model:
  type: RandomForest
  parameters:
    n_estimators: 180
    max_depth: 14
    min_samples_split: 12
    min_samples_leaf: 6
    class_weight: balanced_subsample
```

### Métricas:

- **Accuracy:** 66.65%
- **Precision:** 53.41%
- **Recall:** 64.65% 🥇
- **F1-Score:** 58.49% 🥇
- **ROC-AUC:** 72.53% 🥇

### Justificación de Selección:

1. **Mejor ROC-AUC (72.53%):** Superior capacidad de discriminación entre clientes que harán/no harán churn
2. **Mejor Recall (64.65%):** Detecta el 65% de los clientes en riesgo de abandono
3. **Mejor F1-Score (58.49%):** Mejor balance entre precisión y recall
4. **Impacto de negocio:** En un dataset de 2000 clientes de prueba con ~700 casos de churn, este modelo detecta 453 casos vs 397 del segundo mejor

### Impacto en el Negocio:

Para telecomunicaciones, **detectar clientes en riesgo** es más valioso que accuracy general:

- Un **falso positivo** (ofrecer retención a quien no se irá) tiene bajo costo
- Un **falso negativo** (no detectar a quien se irá) significa perder el cliente

El Experimento 5 minimiza los falsos negativos con el recall más alto.

---

## 📊 Análisis por Métrica

### Mejores por Categoría:

- 🥇 **Accuracy:** Exp 6 (67.5%)
- 🥇 **Precision:** Exp 4 (56.12%)
- 🥇 **Recall:** Exp 5 (64.65%) ⭐
- 🥇 **F1-Score:** Exp 5 (58.49%) ⭐
- 🥇 **ROC-AUC:** Exp 5 (72.53%) ⭐

---

## 🎯 Ranking Final

1. 🥇 **Experimento 5** (Solange) - ROC-AUC: 72.53%
2. 🥈 **Experimento 2** (Nacho) - ROC-AUC: 72.00%
3. 🥉 **Experimento 6** (Solange) - Accuracy: 67.5%
4. **Experimento 3** (Nacho) - Accuracy: 67.15%
5. **Experimento 1** (Nacho) - Baseline sólido
6. **Experimento 4** (Solange) - Recall bajo

---

## 🔬 Aprendizajes Clave

### Configuraciones exitosas:

- **Regularización moderada-alta** funcionó mejor que modelos muy complejos
- **class_weight balanceado** fue crucial para el recall
- **Profundidad limitada** (14-20) evitó overfitting

### Configuraciones menos efectivas:

- Muchos árboles sin regularización → bajo recall
- Profundidad ilimitada → no mejoró significativamente

---

## 📈 Proceso MLOps Utilizado

### Herramientas:

- **Git:** Control de versiones del código
- **GitHub Actions:** CI/CD automático para validar cada experimento
- **DVC:** Versionado de datos y pipeline reproducible
- **MLflow:** Tracking de experimentos y métricas
- **DagsHub:** Plataforma colaborativa para MLOps

### Workflow Implementado:

1. Cada experimento en rama `feat-*` separada
2. Pull Request con descripción detallada de hipótesis y configuración
3. Validación automática mediante CI/CD (GitHub Actions)
4. Revisión colaborativa de métricas
5. Merge del mejor experimento a `main`
6. Cierre documentado de experimentos no seleccionados

### Validación CI/CD:

Cada PR ejecutó automáticamente:
- ✅ Instalación de dependencias
- ✅ Ejecución del pipeline DVC
- ✅ Validación de accuracy mínima (>60%)
- ✅ Registro de métricas y artefactos
- ✅ Tracking en MLflow/DagsHub

---

## 📊 Detalles de Todos los Experimentos

### Experimento 1: RF 500 árboles (Nacho)

**Configuración:**
```
n_estimators: 500
max_depth: 20
min_samples_split: 5
min_samples_leaf: 2
class_weight: balanced_subsample
```

**Resultados:**
- Accuracy: 66.7%
- Recall: 51.03%
- ROC-AUC: 71.00%

**Análisis:** Baseline sólido pero recall insuficiente para negocio.

---

### Experimento 2: RF Regularizado (Nacho) 🥈

**Configuración:**
```
n_estimators: 200
max_depth: 15
min_samples_split: 10
min_samples_leaf: 5
class_weight: balanced_subsample
```

**Resultados:**
- Accuracy: 66.6%
- Recall: 62.59%
- ROC-AUC: 72.00%

**Análisis:** Subcampeón. Excelente balance, podría ser alternativa en producción.

---

### Experimento 3: RF Balanceado (Nacho)

**Configuración:**
```
n_estimators: 250
max_depth: 18
min_samples_split: 4
min_samples_leaf: 3
class_weight: balanced
```

**Resultados:**
- Accuracy: 67.15%
- Recall: 56.81%
- ROC-AUC: 71.60%

**Análisis:** Buena accuracy pero recall medio.

---

### Experimento 4: RF Alto Rendimiento (Solange)

**Configuración:**
```
n_estimators: 450
max_depth: 22
min_samples_split: 3
min_samples_leaf: 1
class_weight: balanced
```

**Resultados:**
- Accuracy: 67.05%
- Recall: 42.92% (el más bajo)
- ROC-AUC: 70.23%

**Análisis:** Muchos árboles sin suficiente regularización resultó en bajo recall.

---

### Experimento 5: RF Conservador (Solange) 🏆

**Configuración:**
```
n_estimators: 180
max_depth: 14
min_samples_split: 12
min_samples_leaf: 6
class_weight: balanced_subsample
```

**Resultados:**
- Accuracy: 66.65%
- Recall: 64.65% 🥇
- ROC-AUC: 72.53% 🥇

**Análisis:** GANADOR. Alta regularización + balance de clases = mejor detección de churn.

---

### Experimento 6: RF Equilibrado (Solange) 🥉

**Configuración:**
```
n_estimators: 320
max_depth: 20
min_samples_split: 5
min_samples_leaf: 3
class_weight: balanced
```

**Resultados:**
- Accuracy: 67.5% 🥇 (la más alta)
- Recall: 56.53%
- ROC-AUC: 71.77%

**Análisis:** Mejor accuracy pero recall insuficiente para el objetivo de negocio.

---

## 👥 Colaboradores

### Nacho
- Rol: Owner del proyecto
- Contribución: 3 experimentos (1, 2, 3)
- Destacado: Experimento 2 (subcampeón)

### Solange
- Rol: Colaboradora
- Contribución: 3 experimentos (4, 5, 6)
- Destacado: Experimento 5 (ganador) 🏆

---

## 📅 Timeline del Proyecto

- **Etapa 1-4:** Setup inicial, pipeline base, experimentos iniciales
- **Etapa 5:** Implementación CI/CD con GitHub Actions
- **Etapa 6:** Iteración colaborativa
  - Primera iteración: Experimentos 1-3 (Nacho)
  - Segunda iteración: Experimentos 4-6 (Solange)
  - Análisis comparativo y selección
  - Merge a main: Experimento 5
- **Estado actual:** Modelo en main listo para producción

---

## 🚀 Próximos Pasos Recomendados

### Inmediato:
1. ✅ Modelo en main con configuración del Exp 5
2. ⏭️ Despliegue a entorno de staging
3. ⏭️ Pruebas con datos recientes

### Corto plazo:
- Implementar monitoreo de métricas en producción
- A/B testing entre Exp 5 y Exp 2
- Dashboard de visualización de predicciones

### Mediano plazo:
- Re-entrenamiento automático mensual
- Incorporar nuevas features
- Análisis de drift de datos

---

## 📚 Documentación Adicional

- **README.md:** Visión general del proyecto
- **GUIA_COLABORADOR.md:** Guía para nuevos colaboradores
- **dvc.yaml:** Definición del pipeline
- **params.yaml:** Configuración del modelo ganador
- **.github/workflows/ci.yaml:** Configuración CI/CD

---

## 🎓 Aprendizajes para el Equipo

### Técnicos:
- La regularización es más importante que la complejidad del modelo
- El balance de clases (class_weight) es crucial en problemas desbalanceados
- Más árboles no siempre es mejor si no hay regularización adecuada

### MLOps:
- CI/CD automatizado acelera la experimentación
- Git branches + PRs facilitan la colaboración
- Documentación clara es esencial para reproducibilidad

### Negocio:
- Las métricas deben alinearse con objetivos de negocio
- En churn, recall > accuracy
- La interpretación de resultados es tan importante como los números

---

## ✅ Conclusión

El proyecto TelcoVision completó exitosamente la **Etapa 6 de Iteración Colaborativa**, generando 6 experimentos validados mediante CI/CD y seleccionando el modelo con mejor desempeño para predicción de churn.

El **Experimento 5** demostró que una configuración conservadora con alta regularización y balance de clases logra los mejores resultados para detectar clientes en riesgo de abandono.

---

**Proyecto:** TelcoVision  
**Versión del Modelo:** Experimento 5 - RF Conservador  
**Fecha de Selección:** Noviembre 2025
