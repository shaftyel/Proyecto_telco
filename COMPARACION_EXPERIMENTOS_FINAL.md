\# 🏆 Comparación Final de Experimentos - Proyecto TelcoVision



\## Resumen Ejecutivo



Se realizaron \*\*6 experimentos\*\* con diferentes configuraciones de Random Forest para optimizar la predicción de churn en telecomunicaciones. El \*\*Experimento 5\*\* fue seleccionado como modelo ganador.



---



\## Tabla Comparativa Completa



| # | Experimento | Autor | Accuracy | Precision | Recall | F1 | ROC-AUC | Estado |

|---|-------------|-------|----------|-----------|--------|-----|---------|--------|

| 1 | RF 500 árboles | Nacho | 66.7% | 54.48% | 51.03% | 52.70% | 71.00% | ❌ Cerrado |

| 2 | RF Regularizado | Nacho | 66.6% | 53.47% | 62.59% | 57.67% | 72.00% | ❌ Cerrado (Subcampeón) |

| 3 | RF Balanceado | Nacho | 67.15% | 54.63% | 56.81% | 55.70% | 71.60% | ❌ Cerrado |

| 4 | RF Alto Rendimiento | Solange | 67.05% | 56.12% | 42.92% | 48.64% | 70.23% | ❌ Cerrado |

| 5 | \*\*RF Conservador\*\* | Solange | 66.65% | 53.41% | \*\*64.65%\*\* | \*\*58.49%\*\* | \*\*72.53%\*\* | ✅ \*\*MERGED\*\* |

| 6 | RF Equilibrado | Solange | \*\*67.5%\*\* | 55.17% | 56.53% | 55.84% | 71.77% | ❌ Cerrado |



---



\## 🏆 Experimento Ganador: #5 - RF Conservador



\### Configuración:

```yaml

model:

&nbsp; type: RandomForest

&nbsp; parameters:

&nbsp;   n\_estimators: 180

&nbsp;   max\_depth: 14

&nbsp;   min\_samples\_split: 12

&nbsp;   min\_samples\_leaf: 6

&nbsp;   class\_weight: balanced\_subsample

```



\### Métricas:

\- \*\*Accuracy:\*\* 66.65%

\- \*\*Precision:\*\* 53.41%

\- \*\*Recall:\*\* 64.65% 🥇

\- \*\*F1-Score:\*\* 58.49% 🥇

\- \*\*ROC-AUC:\*\* 72.53% 🥇



\### Justificación de Selección:



1\. \*\*Mejor ROC-AUC (72.53%):\*\* Superior capacidad de discriminación entre clientes que harán/no harán churn

2\. \*\*Mejor Recall (64.65%):\*\* Detecta el 65% de los clientes en riesgo de abandono

3\. \*\*Mejor F1-Score (58.49%):\*\* Mejor balance entre precisión y recall

4\. \*\*Impacto de negocio:\*\* En un dataset de 2000 clientes de prueba con ~700 casos de churn, este modelo detecta 453 casos vs 397 del segundo mejor



\### Impacto en el Negocio:



Para telecomunicaciones, \*\*detectar clientes en riesgo\*\* es más valioso que accuracy general:

\- Un \*\*falso positivo\*\* (ofrecer retención a quien no se irá) tiene bajo costo

\- Un \*\*falso negativo\*\* (no detectar a quien se irá) significa perder el cliente



El Experimento 5 minimiza los falsos negativos con el recall más alto.



---



\## 📊 Análisis por Métrica



\### Mejores por Categoría:

\- 🥇 \*\*Accuracy:\*\* Exp 6 (67.5%)

\- 🥇 \*\*Precision:\*\* Exp 4 (56.12%)

\- 🥇 \*\*Recall:\*\* Exp 5 (64.65%) ⭐

\- 🥇 \*\*F1-Score:\*\* Exp 5 (58.49%) ⭐

\- 🥇 \*\*ROC-AUC:\*\* Exp 5 (72.53%) ⭐



---



\## 🎯 Ranking Final



1\. 🥇 \*\*Experimento 5\*\* (Solange) - ROC-AUC: 72.53%

2\. 🥈 \*\*Experimento 2\*\* (Nacho) - ROC-AUC: 72.00%

3\. 🥉 \*\*Experimento 6\*\* (Solange) - Accuracy: 67.5%

4\. \*\*Experimento 3\*\* (Nacho) - Accuracy: 67.15%

5\. \*\*Experimento 1\*\* (Nacho) - Baseline sólido

6\. \*\*Experimento 4\*\* (Solange) - Recall bajo



---



\## 🔬 Aprendizajes Clave



\### Configuraciones exitosas:

\- \*\*Regularización moderada-alta\*\* funcionó mejor que modelos muy complejos

\- \*\*class\_weight balanceado\*\* fue crucial para el recall

\- \*\*Profundidad limitada\*\* (14-20) evitó overfitting



\### Configuraciones menos efectivas:

\- Muchos árboles sin regularización → bajo recall

\- Profundidad ilimitada → no mejoró significativamente



---



\## 📈 Proceso MLOps Utilizado



\### Herramientas:

\- \*\*Git:\*\* Control de versiones

\- \*\*GitHub Actions:\*\* CI/CD automático

\- \*\*DVC:\*\* Versionado de datos y modelos

\- \*\*MLflow:\*\* Tracking de experimentos

\- \*\*DagsHub:\*\* Colaboración y visualización



\### Workflow:

1\. Cada experimento en rama `feat-\*` separada

2\. Pull Request con descripción detallada

3\. Validación automática de CI/CD

4\. Revisión de métricas

5\. Merge del mejor a `main`



---



\## 👥 Colaboradores



\- \*\*Nacho:\*\* 3 experimentos (baseline, regularizado, balanceado)

\- \*\*Solange:\*\* 3 experimentos (alto rendimiento, conservador, equilibrado)



---



\## 📅 Timeline



\- \*\*Experimentos 1-3:\*\* Primera iteración (Nacho)

\- \*\*Experimentos 4-6:\*\* Segunda iteración colaborativa (Solange)

\- \*\*Selección:\*\* Experimento 5 tras análisis comparativo

\- \*\*Merge a main:\*\* \[Fecha actual]



---



\## 🚀 Próximos Pasos



1\. ✅ Modelo en producción con configuración del Exp 5

2\. ⏭️ Monitoreo continuo de métricas en datos reales

3\. ⏭️ A/B testing con Exp 2 como alternativa

4\. ⏭️ Re-entrenamiento mensual con datos nuevos



---



\*\*Proyecto:\*\* TelcoVision  

\*\*Fecha:\*\* Noviembre 2025  

\*\*Etapa:\*\* 6 - Iteración Colaborativa ✅ COMPLETADA

