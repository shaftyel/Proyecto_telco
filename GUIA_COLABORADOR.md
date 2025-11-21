# 📘 Guía para Colaboradores - Proyecto TelcoVision

## **Bienvenido al Proyecto**

Esta guía te ayudará a configurar el entorno y realizar experimentos en el proyecto **TelcoVision**, un sistema de predicción de churn para telecomunicaciones que utiliza MLOps moderno con DVC, MLflow, GitHub Actions y DagsHub.

**🔐 Importante:** Esta guía incluye credenciales de acceso al proyecto. Son para uso exclusivo del equipo académico. Por favor, no las compartas fuera del contexto del proyecto.

---

## **📋 Requisitos Previos**

- Anaconda o Miniconda instalado
- Git instalado
- Acceso a GitHub
- Credenciales del proyecto (ver abajo)

---

## **🔑 Credenciales del Proyecto**

**Usa estas credenciales durante la configuración:**

| Recurso | Valor |
|---------|-------|
| **GitHub Repository** | `https://github.com/Shaftyel/Proyecto_telco.git` |
| **DagsHub Repository** | `https://dagshub.com/Nacho/proyecto_telco.git` |
| **DagsHub Usuario** | `Nacho` |
| **DagsHub Token** | `************` |

**⚠️ Nota de seguridad:** Estas credenciales son para propósitos académicos del proyecto. No las compartas fuera del equipo.

---

## **⚡ Inicio Rápido (Comandos Clave)**

Si ya estás familiarizado con Git y conda, estos son los comandos esenciales:

```bash
# 1. Clonar y configurar
git clone https://github.com/Shaftyel/Proyecto_telco.git
cd Proyecto_telco
conda create -n telcovision_colab python=3.9 -y
conda activate telcovision_colab
pip install -r requirements.txt

# 2. Configurar remotes
git remote add dagshub https://dagshub.com/Nacho/proyecto_telco.git

# 3. Configurar DVC
dvc remote modify dagshub auth basic
dvc remote modify dagshub user Nacho
dvc remote modify dagshub password COLOCAR_TOKEN

# 4. Ejecutar pipeline
dvc repro

# 5. Crear experimento
git checkout -b feat-experimento-[TU_NOMBRE]
notepad params.yaml  # Modificar parámetros
dvc repro
git add params.yaml dvc.lock
git commit -m "feat: experimento [TU_NOMBRE]"
git push origin feat-experimento-[TU_NOMBRE]
# Luego crear PR en GitHub
```

Para instrucciones detalladas, continúa leyendo.

---

# 🚀 PARTE 1: Configuración Inicial

## **Paso 1: Clonar el repositorio**

Abre **Anaconda Prompt** y ejecuta:

```bash
# Navegar a donde quieras tener el proyecto
cd C:\Users\TU_USUARIO\Documents

# Clonar desde GitHub
git clone https://github.com/Shaftyel/Proyecto_telco.git

# Entrar al proyecto
cd Proyecto_telco
```

---

## **Paso 2: Crear y activar entorno conda**

```bash
# Crear entorno con Python 3.9
conda create -n telcovision_colab python=3.9 -y

# Activar entorno
conda activate telcovision_colab

# Instalar dependencias
pip install -r requirements.txt
```

**⏱️ Tiempo estimado:** 5-10 minutos

---

## **Paso 3: Configurar Git remotes**

```bash
# Ver remotes actuales
git remote -v

# Deberías ver solo 'origin' apuntando a GitHub

# Agregar DagsHub como segundo remote
git remote add dagshub https://dagshub.com/Nacho/proyecto_telco.git

# Verificar que ambos están configurados
git remote -v
```

**Resultado esperado:**
```
origin     https://github.com/Shaftyel/Proyecto_telco.git (fetch)
origin     https://github.com/Shaftyel/Proyecto_telco.git (push)
dagshub   https://dagshub.com/Nacho/proyecto_telco.git (fetch)
dagshub   https://dagshub.com/Nacho/proyecto_telco.git (push)
```

---

## **Paso 4: Configurar credenciales DVC**

```bash
# Configurar autenticación DVC con DagsHub
dvc remote modify dagshub auth basic
dvc remote modify dagshub user Nacho
dvc remote modify dagshub password COLOCAR_TOKEN
```

**✅ Las credenciales ya están incluidas en los comandos de arriba.**

---

## **Paso 5: Obtener los datos y ejecutar pipeline**

```bash
# Verificar que existen los datos raw
dir data\raw

# Deberías ver: telco_churn.csv

# Ejecutar pipeline completo para generar todo
dvc repro
```

**Esto ejecutará:**
1. `data_prep` - Preprocesamiento de datos
2. `train` - Entrenamiento del modelo

**⏱️ Tiempo estimado:** 2-5 minutos (depende de tu máquina)

---

## **Paso 6: Verificar que todo funciona**

```bash
# Ver las métricas del modelo baseline
type models\metrics.json
```

**Deberías ver algo como:**
```json
{
  "accuracy": 0.67,
  "precision": 0.54,
  "recall": 0.57,
  "f1": 0.56,
  "roc_auc": 0.72
}
```

✅ **Si ves las métricas, ¡todo está funcionando correctamente!**

---

# 🧪 PARTE 2: Crear tu Experimento

## **Paso 7: Actualizar desde main**

Antes de empezar tu experimento, asegúrate de tener la última versión:

```bash
# Asegurarse de estar en main
git checkout main

# Actualizar desde GitHub
git pull origin main
```

---

## **Paso 8: Crear rama para tu experimento**

```bash
# Crear rama con nombre descriptivo
# Usa tu nombre o iniciales para identificarla
git checkout -b feat-experimento-[TU_NOMBRE]

# Ejemplo:
# git checkout -b feat-experimento-juan

# Verificar que estás en la nueva rama
git branch
```

El asterisco (*) debe estar junto a tu nueva rama.

---

## **Paso 9: Modificar parámetros del modelo**

```bash
# Abrir el archivo de configuración
notepad params.yaml
```

### **Estructura del archivo:**

```yaml
### params.yaml
paths:
  processed_data: data/processed/telco_churn_processed.csv
  model_path: models/model.joblib
  metrics_path: models/metrics.json

target: churn
test_size: 0.2
random_state: 42

model:
  type: RandomForest
  parameters:
    n_estimators: 300      # Número de árboles
    max_depth: null        # Profundidad máxima (null = sin límite)
    min_samples_split: 2   # Mínimo de muestras para dividir
    min_samples_leaf: 1    # Mínimo de muestras por hoja
    class_weight: balanced_subsample  # Manejo de clases desbalanceadas
```

### **Ideas para experimentos:**

**Experimento 1: Más árboles con profundidad controlada**
```yaml
model:
  type: RandomForest
  parameters:
    n_estimators: 400
    max_depth: 20
    min_samples_split: 3
    min_samples_leaf: 2
    class_weight: balanced
```

**Experimento 2: Modelo más regularizado**
```yaml
model:
  type: RandomForest
  parameters:
    n_estimators: 150
    max_depth: 12
    min_samples_split: 15
    min_samples_leaf: 8
    class_weight: balanced_subsample
```

**Experimento 3: Optimización balanceada**
```yaml
model:
  type: RandomForest
  parameters:
    n_estimators: 350
    max_depth: 25
    min_samples_split: 4
    min_samples_leaf: 3
    class_weight: balanced
```

### **Explicación de parámetros:**

- `n_estimators`: Más árboles = más precisión pero más lento
- `max_depth`: Profundidad máxima de cada árbol (null = sin límite)
- `min_samples_split`: Mínimo de muestras para dividir un nodo (mayor = más regularización)
- `min_samples_leaf`: Mínimo de muestras en cada hoja (mayor = más regularización)
- `class_weight`: 
  - `balanced`: Ajusta pesos para balancear clases
  - `balanced_subsample`: Similar pero con submuestreo en cada árbol

**⚠️ IMPORTANTE: Guardar con Ctrl+S antes de cerrar Notepad**

---

## **Paso 10: Ejecutar tu experimento**

```bash
# Ejecutar el pipeline con tus nuevos parámetros
dvc repro

# Ver las métricas obtenidas
type models\metrics.json
```

### **Anotar tus resultados:**

```
Accuracy:  _____%
Precision: _____%
Recall:    _____%
F1-Score:  _____%
ROC-AUC:   _____%
```

**💡 Tip:** Copia el contenido completo de `metrics.json` para incluirlo en tu PR.

---

## **Paso 11: Commitear tus cambios**

```bash
# Ver qué archivos cambiaron
git status

# Deberías ver:
# - params.yaml (modificado)
# - dvc.lock (modificado)

# Agregar archivos al staging
git add params.yaml dvc.lock

# Crear commit con mensaje descriptivo
git commit -m "feat: experimento [TU_NOMBRE] - RF optimizado"

# Ejemplo:
# git commit -m "feat: experimento juan - RF 400 árboles depth 20"
```

---

## **Paso 12: Pushear a GitHub**

```bash
# Subir tu rama a GitHub
git push origin feat-experimento-[TU_NOMBRE]
```

**Ejemplo de output exitoso:**
```
Enumerating objects: 7, done.
Counting objects: 100% (7/7), done.
...
To https://github.com/Shaftyel/Proyecto_telco.git
 * [new branch]      feat-experimento-juan -> feat-experimento-juan
```

---

## **Paso 13: Crear Pull Request en GitHub**

### **En el navegador:**

1. Ve a: `https://github.com/Shaftyel/Proyecto_telco`

2. Verás un banner amarillo: **"feat-experimento-[TU_NOMBRE] had recent pushes"**

3. Click en el botón verde **"Compare & pull request"**

4. Llenar la información del PR:

**Título:**
```
Experimento [TU_NOMBRE]: Random Forest Optimizado
```

**Descripción:**
```markdown
## 🎯 Experimento: Random Forest Optimizado

### Autor: [Tu Nombre]

### Cambios realizados:
- `n_estimators`: 300 → [TU_VALOR]
- `max_depth`: null → [TU_VALOR]
- `min_samples_split`: 2 → [TU_VALOR]
- `min_samples_leaf`: 1 → [TU_VALOR]
- `class_weight`: balanced_subsample → [TU_VALOR]

### Hipótesis:
[Explica por qué elegiste estos parámetros y qué esperas lograr]

### Resultados obtenidos:
```json
[Pega aquí el contenido de models/metrics.json]
```

### Observaciones:
[Cualquier observación sobre el entrenamiento, tiempo de ejecución, etc.]

cc @Shaftyel para revisión
```

5. Click en **"Create pull request"**

---

## **Paso 14: Esperar validación automática**

GitHub Actions ejecutará automáticamente:
- ✅ Instalación de dependencias
- ✅ Ejecución del pipeline con tus parámetros
- ✅ Validación de métricas (accuracy > 60%)
- ✅ Guardado de artefactos del modelo

**Verás en el PR:**
- ⏳ Círculo amarillo = Ejecutando
- ✅ Check verde = ¡Todo correcto!
- ❌ X roja = Algo falló (revisa los logs)

---

# 🔄 PARTE 3: Hacer Más Experimentos (Opcional)

Si quieres probar otra configuración:

```bash
# Volver a main
git checkout main
git pull origin main

# Crear nueva rama
git checkout -b feat-experimento-[TU_NOMBRE]-v2

# Editar params.yaml con nuevos valores
notepad params.yaml

# Ejecutar, commitear y crear PR
dvc repro
type models\metrics.json
git add params.yaml dvc.lock
git commit -m "feat: experimento [TU_NOMBRE] v2 - [descripción]"
git push origin feat-experimento-[TU_NOMBRE]-v2
```

Luego crea otro PR en GitHub.

---

# 📊 PARTE 4: Comparar con Otros Experimentos

### **Experimentos existentes en el proyecto:**

| Experimento | Accuracy | Precision | Recall | F1 | ROC-AUC | Autor |
|-------------|----------|-----------|--------|-----|---------|-------|
| **Exp 1: 500 árboles** | 66.7% | 54.5% | 51.0% | 52.7% | 71.0% | Nacho |
| **Exp 2: Regularizado** | 66.6% | 53.5% | **62.6%** ⭐ | **57.7%** ⭐ | **72.0%** ⭐ | Nacho |
| **Exp 3: Balanceado** | **67.2%** ⭐ | **54.6%** ⭐ | 56.8% | 55.7% | 71.6% | Nacho |
| **Tu experimento** | ?% | ?% | ?% | ?% | ?% | Tú |

**Mejores valores actuales:**
- 🥇 **Mejor Accuracy:** 67.2% (Exp 3)
- 🥇 **Mejor Recall:** 62.6% (Exp 2) - Detecta más casos de churn
- 🥇 **Mejor ROC-AUC:** 72.0% (Exp 2) - Mejor discriminación general

### **¿Qué buscar?**

Para un problema de **predicción de churn**:
- **Recall alto** = Detectamos más clientes que van a abandonar
- **ROC-AUC alto** = Mejor capacidad de discriminación general
- **F1 alto** = Buen balance entre precisión y recall

---

# 📋 Checklist Final

- [ ] Repositorio clonado correctamente
- [ ] Entorno conda creado y activado
- [ ] Dependencias instaladas sin errores
- [ ] DVC configurado con credenciales
- [ ] Pipeline ejecutado exitosamente (dvc repro)
- [ ] Rama de experimento creada
- [ ] Parámetros modificados en params.yaml
- [ ] Experimento ejecutado y métricas obtenidas
- [ ] Cambios commitados (git add, git commit)
- [ ] Push a GitHub exitoso
- [ ] Pull Request creado con descripción completa
- [ ] CI/CD pasando en verde ✅

---

# 🆘 Troubleshooting

## **Problema: "ERROR: failed to reproduce 'train': output is already tracked by SCM"**

**Solución:**
```bash
git rm --cached models/metrics.json
git rm --cached models/model.joblib
git commit -m "fix: remover outputs del pipeline de Git"
git push origin feat-experimento-[TU_NOMBRE] --force
```

---

## **Problema: "dvc pull" falla**

**Solución:**
Los datos raw están temporalmente en Git, así que simplemente ejecuta:
```bash
dvc repro
```

---

## **Problema: Falta alguna dependencia de Python**

**Solución:**
```bash
pip install [nombre-paquete] --break-system-packages
```

---

## **Problema: "fatal: not a git repository"**

**Solución:**
No estás en la carpeta del proyecto. Navega correctamente:
```bash
cd C:\Users\TU_USUARIO\Documents\Proyecto_telco
```

---

## **Problema: El workflow de CI falla en GitHub Actions**

**Pasos a seguir:**
1. Ve al PR en GitHub
2. Click en "Details" junto al check fallido
3. Lee los logs para identificar el error
4. Si no lo entiendes, copia el error y contacta al owner

---

# 📞 Contacto y Soporte

**Owner del proyecto:** @Shaftyel (Nacho)

**Recursos del proyecto:**
- **GitHub:** https://github.com/Shaftyel/Proyecto_telco
- **DagsHub:** https://dagshub.com/Nacho/proyecto_telco
- **DagsHub MLflow:** https://dagshub.com/Nacho/proyecto_telco.mlflow

**Credenciales (recordatorio):**
- Usuario DagsHub: `Nacho`
- Token DagsHub: `********`

**Para dudas:**
- Comenta directamente en tu Pull Request
- Menciona a @Shaftyel con `@` en el comentario

---

# 🎯 Métricas de Éxito

Tu experimento será considerado exitoso si:
- ✅ CI/CD pasa en verde
- ✅ Accuracy > 60%
- ✅ El experimento aporta insights sobre los parámetros probados
- ✅ La documentación en el PR es clara y completa

**No te preocupes si tu modelo no es el "mejor"** - el objetivo es aprender sobre el proceso MLOps colaborativo y cómo diferentes configuraciones afectan el rendimiento.

---

# 🚀 Próximos Pasos

Después de que todos los colaboradores hayan hecho sus experimentos:

1. El owner comparará TODOS los resultados
2. Se seleccionará el mejor modelo
3. Se hará merge a `main`
4. Los demás PRs se cerrarán con justificación documentada

**¡Gracias por contribuir al proyecto TelcoVision!** 🎉

---

**Versión:** 1.0  
**Última actualización:** Noviembre 2025  
**Autor:** Proyecto TelcoVision Team
