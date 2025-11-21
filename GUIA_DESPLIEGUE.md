# 🚀 Guía de Despliegue - TelcoVision

## Descripción General

Este documento describe cómo desplegar el modelo de predicción de churn de TelcoVision en un entorno de producción usando diferentes tecnologías.

---

## 📋 Requisitos Previos

- Python 3.9+
- Modelo entrenado en `models/model.joblib`
- Dependencias del proyecto instaladas
- Datos procesados disponibles

---

## 🎯 Opciones de Despliegue

### Opción 1: API REST con FastAPI (Recomendado)
### Opción 2: Aplicación Web con Streamlit
### Opción 3: Batch Processing
### Opción 4: Contenedor Docker

---

# 🔷 Opción 1: API REST con FastAPI

## **¿Por qué FastAPI?**

- ✅ Alto rendimiento (comparable con Node.js y Go)
- ✅ Documentación automática (Swagger/OpenAPI)
- ✅ Validación automática de datos con Pydantic
- ✅ Fácil integración con sistemas empresariales
- ✅ Soporte para async/await

---

## **Paso 1: Instalar dependencias adicionales**

```bash
pip install fastapi uvicorn pydantic python-multipart
```

Agregar a `requirements.txt`:
```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

---

## **Paso 2: Crear script de API (`src/api.py`)**

```python
"""
API REST para predicción de churn - TelcoVision
Desplegable con FastAPI y uvicorn
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import yaml

# Cargar configuración
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Cargar modelo al iniciar la API
model = joblib.load(params['paths']['model_path'])

# Crear app FastAPI
app = FastAPI(
    title="TelcoVision Churn Prediction API",
    description="API para predicción de churn en telecomunicaciones",
    version="1.0.0"
)


class CustomerData(BaseModel):
    """Schema de entrada para un cliente"""
    # Agregar aquí todos los features necesarios según tu dataset
    # Ejemplo (ajustar según tus features reales):
    tenure: float = Field(..., description="Meses de antigüedad")
    monthly_charges: float = Field(..., description="Cargo mensual")
    total_charges: float = Field(..., description="Cargo total acumulado")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12.0,
                "monthly_charges": 70.5,
                "total_charges": 846.0
            }
        }


class PredictionResponse(BaseModel):
    """Schema de respuesta"""
    churn_probability: float
    will_churn: bool
    risk_level: str
    confidence: float


@app.get("/")
def root():
    """Endpoint raíz"""
    return {
        "message": "TelcoVision Churn Prediction API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
def health_check():
    """Health check para monitoreo"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predecir probabilidad de churn para un cliente
    
    Args:
        customer: Datos del cliente
        
    Returns:
        Probabilidad de churn y nivel de riesgo
    """
    try:
        # Convertir a DataFrame
        data = pd.DataFrame([customer.dict()])
        
        # Predecir
        churn_proba = model.predict_proba(data)[0, 1]
        will_churn = bool(churn_proba >= 0.5)
        
        # Determinar nivel de riesgo
        if churn_proba < 0.3:
            risk_level = "Bajo"
        elif churn_proba < 0.6:
            risk_level = "Medio"
        else:
            risk_level = "Alto"
        
        # Confidence (qué tan seguro está el modelo)
        confidence = max(churn_proba, 1 - churn_proba)
        
        return PredictionResponse(
            churn_probability=float(churn_proba),
            will_churn=will_churn,
            risk_level=risk_level,
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict_batch")
def predict_batch(customers: List[CustomerData]):
    """
    Predecir churn para múltiples clientes
    
    Args:
        customers: Lista de clientes
        
    Returns:
        Lista de predicciones
    """
    try:
        results = []
        for customer in customers:
            pred = predict_churn(customer)
            results.append(pred.dict())
        
        return {
            "total_customers": len(customers),
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en batch: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## **Paso 3: Ejecutar la API**

```bash
# Ejecutar en desarrollo
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# O con python directamente
python src/api.py
```

Acceder a:
- API: `http://localhost:8000`
- Documentación Swagger: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## **Paso 4: Probar la API**

**Usando curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 70.5, "total_charges": 846.0}'
```

**Usando Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "tenure": 12.0,
    "monthly_charges": 70.5,
    "total_charges": 846.0
}

response = requests.post(url, json=data)
print(response.json())
```

---

## **Paso 5: Despliegue en producción**

### **Opción A: Servidor con systemd**

Crear servicio en `/etc/systemd/system/telcovision.service`:

```ini
[Unit]
Description=TelcoVision Churn Prediction API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/telcovision
Environment="PATH=/opt/telcovision/venv/bin"
ExecStart=/opt/telcovision/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
```

Activar:
```bash
sudo systemctl enable telcovision
sudo systemctl start telcovision
sudo systemctl status telcovision
```

---

### **Opción B: Docker**

Crear `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Construir y ejecutar:
```bash
docker build -t telcovision-api .
docker run -p 8000:8000 telcovision-api
```

---

### **Opción C: Servicios Cloud**

**AWS (Elastic Beanstalk):**
```bash
eb init -p python-3.9 telcovision-api
eb create telcovision-env
eb deploy
```

**Google Cloud (Cloud Run):**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/telcovision
gcloud run deploy --image gcr.io/PROJECT_ID/telcovision --platform managed
```

**Azure (App Service):**
```bash
az webapp up --name telcovision --runtime "PYTHON:3.9"
```

---

# 🎨 Opción 2: Aplicación Web con Streamlit

## **¿Cuándo usar Streamlit?**

- ✅ Demos y prototipos rápidos
- ✅ Dashboards internos
- ✅ Aplicaciones para data scientists
- ✅ No requiere conocimientos de frontend

---

## **Paso 1: Instalar Streamlit**

```bash
pip install streamlit plotly
```

---

## **Paso 2: Crear app (`src/app_streamlit.py`)**

```python
"""
Aplicación Streamlit para predicción de churn - TelcoVision
"""

import streamlit as st
import joblib
import pandas as pd
import yaml
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(
    page_title="TelcoVision - Predicción de Churn",
    page_icon="📊",
    layout="wide"
)

# Cargar modelo
@st.cache_resource
def load_model():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return joblib.load(params['paths']['model_path'])

model = load_model()

# Título
st.title("📊 TelcoVision - Predicción de Churn")
st.markdown("Sistema de predicción de abandono de clientes en telecomunicaciones")

# Sidebar con inputs
st.sidebar.header("Datos del Cliente")

# Inputs (ajustar según tus features reales)
tenure = st.sidebar.number_input("Meses de antigüedad", min_value=0, max_value=120, value=12)
monthly_charges = st.sidebar.number_input("Cargo mensual ($)", min_value=0.0, max_value=200.0, value=70.5)
total_charges = st.sidebar.number_input("Cargo total ($)", min_value=0.0, max_value=10000.0, value=846.0)

# Botón de predicción
if st.sidebar.button("Predecir Churn", type="primary"):
    # Crear DataFrame
    data = pd.DataFrame({
        'tenure': [tenure],
        'monthly_charges': [monthly_charges],
        'total_charges': [total_charges]
    })
    
    # Predecir
    churn_proba = model.predict_proba(data)[0, 1]
    will_churn = churn_proba >= 0.5
    
    # Mostrar resultados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probabilidad de Churn", f"{churn_proba*100:.1f}%")
    
    with col2:
        st.metric("Predicción", "CHURN ❌" if will_churn else "NO CHURN ✅")
    
    with col3:
        if churn_proba < 0.3:
            risk = "Bajo 🟢"
        elif churn_proba < 0.6:
            risk = "Medio 🟡"
        else:
            risk = "Alto 🔴"
        st.metric("Nivel de Riesgo", risk)
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = churn_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Churn (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if churn_proba > 0.6 else "orange" if churn_proba > 0.3 else "green"},
            'steps' : [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones
    st.subheader("📋 Recomendaciones")
    if will_churn:
        st.warning("""
        **Acciones sugeridas:**
        - Contactar al cliente proactivamente
        - Ofrecer descuentos o promociones personalizadas
        - Revisar experiencia del cliente y satisfacción
        - Evaluar calidad de servicio
        """)
    else:
        st.success("""
        **Cliente con baja probabilidad de churn:**
        - Mantener calidad de servicio actual
        - Considerar programas de fidelización
        - Monitorear satisfacción periódicamente
        """)
```

---

## **Paso 3: Ejecutar Streamlit**

```bash
streamlit run src/app_streamlit.py
```

Acceder a: `http://localhost:8501`

---

## **Paso 4: Desplegar Streamlit**

**Streamlit Cloud (gratis):**
1. Pushear código a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar repositorio
4. Seleccionar `src/app_streamlit.py`
5. Deploy

---

# 📦 Opción 3: Procesamiento por Lotes (Batch)

Para procesar grandes volúmenes de clientes de una vez.

## **Script de batch (`src/batch_predict.py`)**

```python
"""
Predicción batch para múltiples clientes
"""

import joblib
import pandas as pd
import yaml
from datetime import datetime
import os

def batch_predict(input_file: str, output_file: str):
    """
    Predecir churn para un archivo CSV de clientes
    
    Args:
        input_file: CSV con datos de clientes
        output_file: CSV de salida con predicciones
    """
    # Cargar modelo
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    model = joblib.load(params['paths']['model_path'])
    
    # Cargar datos
    df = pd.read_csv(input_file)
    print(f"Procesando {len(df)} clientes...")
    
    # Predecir
    df['churn_probability'] = model.predict_proba(df)[:, 1]
    df['will_churn'] = df['churn_probability'] >= 0.5
    df['risk_level'] = pd.cut(
        df['churn_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # Guardar
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Resultados guardados en: {output_file}")
    
    # Resumen
    print(f"\nResumen:")
    print(f"- Total clientes: {len(df)}")
    print(f"- Churn predicho: {df['will_churn'].sum()} ({df['will_churn'].mean()*100:.1f}%)")
    print(f"- Riesgo Alto: {(df['risk_level']=='Alto').sum()}")
    print(f"- Riesgo Medio: {(df['risk_level']=='Medio').sum()}")
    print(f"- Riesgo Bajo: {(df['risk_level']=='Bajo').sum()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV de entrada')
    parser.add_argument('--output', required=True, help='CSV de salida')
    args = parser.parse_args()
    
    batch_predict(args.input, args.output)
```

**Uso:**
```bash
python src/batch_predict.py \
  --input data/clientes_nuevos.csv \
  --output predictions/batch_20251120.csv
```

---

# 🐳 Opción 4: Despliegue con Docker Compose

Para desplegar API + Base de Datos + Monitoring.

## **`docker-compose.yml`**

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.joblib
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:14
    environment:
      POSTGRES_DB: telcovision
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  postgres_data:
```

Ejecutar:
```bash
docker-compose up -d
```

---

# 📊 Monitoreo y Mantenimiento

## **Métricas a Monitorear**

1. **Performance del Modelo:**
   - Accuracy, Precision, Recall en producción
   - Data drift (cambios en distribución de datos)
   - Prediction drift (cambios en distribución de predicciones)

2. **Performance del Sistema:**
   - Latencia de respuesta (target: <100ms)
   - Throughput (requests/segundo)
   - Uso de memoria y CPU
   - Tasa de errores

3. **Negocio:**
   - Tasa de churn real vs predicho
   - ROI de acciones preventivas
   - Costo de falsos positivos/negativos

## **Re-entrenamiento**

Configurar pipeline automático:
```bash
# Cron job mensual
0 0 1 * * cd /opt/telcovision && dvc repro && git add . && git commit -m "Reentrenamiento mensual" && dvc push
```

---

# 🔒 Seguridad

## **Buenas Prácticas**

1. **Autenticación:**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
def predict(customer: CustomerData, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validar token
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    # ... predicción
```

2. **Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
def predict(request: Request, customer: CustomerData):
    # ...
```

3. **HTTPS:**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
```

---

# 📝 Checklist de Despliegue

- [ ] Modelo entrenado y validado
- [ ] API implementada y probada localmente
- [ ] Tests de integración pasando
- [ ] Documentación actualizada
- [ ] Variables de entorno configuradas
- [ ] Logging implementado
- [ ] Monitoreo configurado
- [ ] Backup automatizado
- [ ] Plan de rollback definido
- [ ] Equipo capacitado

---

# 🎓 Recursos Adicionales

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Docker Documentation](https://docs.docker.com)
- [MLOps Best Practices](https://ml-ops.org)

---

**Versión:** 1.0  
**Última actualización:** Noviembre 2025  
**Proyecto:** TelcoVision
