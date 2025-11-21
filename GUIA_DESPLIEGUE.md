\# 🚀 Guía de Despliegue - TelcoVision



\## Descripción General



Este documento describe cómo desplegar el modelo de predicción de churn de TelcoVision en un entorno de producción usando diferentes tecnologías.



---



\## 📋 Requisitos Previos



\- Python 3.9+

\- Modelo entrenado en `models/model.joblib`

\- Dependencias del proyecto instaladas

\- Datos procesados disponibles



---



\## 🎯 Opciones de Despliegue



\### Opción 1: API REST con FastAPI (Recomendado)

\### Opción 2: Aplicación Web con Streamlit

\### Opción 3: Batch Processing

\### Opción 4: Contenedor Docker



---



\# 🔷 Opción 1: API REST con FastAPI



\## \*\*¿Por qué FastAPI?\*\*



\- ✅ Alto rendimiento (comparable con Node.js y Go)

\- ✅ Documentación automática (Swagger/OpenAPI)

\- ✅ Validación automática de datos con Pydantic

\- ✅ Fácil integración con sistemas empresariales

\- ✅ Soporte para async/await



---



\## \*\*Paso 1: Instalar dependencias adicionales\*\*

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



\## \*\*Paso 2: Crear script de API (`src/api.py`)\*\*

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



\# Cargar configuración

with open('params.yaml', 'r') as f:

&nbsp;   params = yaml.safe\_load(f)



\# Cargar modelo al iniciar la API

model = joblib.load(params\['paths']\['model\_path'])



\# Crear app FastAPI

app = FastAPI(

&nbsp;   title="TelcoVision Churn Prediction API",

&nbsp;   description="API para predicción de churn en telecomunicaciones",

&nbsp;   version="1.0.0"

)





class CustomerData(BaseModel):

&nbsp;   """Schema de entrada para un cliente"""

&nbsp;   # Agregar aquí todos los features necesarios según tu dataset

&nbsp;   # Ejemplo (ajustar según tus features reales):

&nbsp;   tenure: float = Field(..., description="Meses de antigüedad")

&nbsp;   monthly\_charges: float = Field(..., description="Cargo mensual")

&nbsp;   total\_charges: float = Field(..., description="Cargo total acumulado")

&nbsp;   # ... agregar resto de features

&nbsp;   

&nbsp;   class Config:

&nbsp;       json\_schema\_extra = {

&nbsp;           "example": {

&nbsp;               "tenure": 12.0,

&nbsp;               "monthly\_charges": 70.5,

&nbsp;               "total\_charges": 846.0

&nbsp;           }

&nbsp;       }





class PredictionResponse(BaseModel):

&nbsp;   """Schema de respuesta"""

&nbsp;   churn\_probability: float

&nbsp;   will\_churn: bool

&nbsp;   risk\_level: str

&nbsp;   confidence: float





@app.get("/")

def root():

&nbsp;   """Endpoint raíz"""

&nbsp;   return {

&nbsp;       "message": "TelcoVision Churn Prediction API",

&nbsp;       "version": "1.0.0",

&nbsp;       "status": "active"

&nbsp;   }





@app.get("/health")

def health\_check():

&nbsp;   """Health check para monitoreo"""

&nbsp;   return {

&nbsp;       "status": "healthy",

&nbsp;       "model\_loaded": model is not None

&nbsp;   }





@app.post("/predict", response\_model=PredictionResponse)

def predict\_churn(customer: CustomerData):

&nbsp;   """

&nbsp;   Predecir probabilidad de churn para un cliente

&nbsp;   

&nbsp;   Args:

&nbsp;       customer: Datos del cliente

&nbsp;       

&nbsp;   Returns:

&nbsp;       Probabilidad de churn y nivel de riesgo

&nbsp;   """

&nbsp;   try:

&nbsp;       # Convertir a DataFrame

&nbsp;       data = pd.DataFrame(\[customer.dict()])

&nbsp;       

&nbsp;       # Predecir

&nbsp;       churn\_proba = model.predict\_proba(data)\[0, 1]

&nbsp;       will\_churn = bool(churn\_proba >= 0.5)

&nbsp;       

&nbsp;       # Determinar nivel de riesgo

&nbsp;       if churn\_proba < 0.3:

&nbsp;           risk\_level = "Bajo"

&nbsp;       elif churn\_proba < 0.6:

&nbsp;           risk\_level = "Medio"

&nbsp;       else:

&nbsp;           risk\_level = "Alto"

&nbsp;       

&nbsp;       # Confidence (qué tan seguro está el modelo)

&nbsp;       confidence = max(churn\_proba, 1 - churn\_proba)

&nbsp;       

&nbsp;       return PredictionResponse(

&nbsp;           churn\_probability=float(churn\_proba),

&nbsp;           will\_churn=will\_churn,

&nbsp;           risk\_level=risk\_level,

&nbsp;           confidence=float(confidence)

&nbsp;       )

&nbsp;       

&nbsp;   except Exception as e:

&nbsp;       raise HTTPException(status\_code=500, detail=f"Error en predicción: {str(e)}")





@app.post("/predict\_batch")

def predict\_batch(customers: List\[CustomerData]):

&nbsp;   """

&nbsp;   Predecir churn para múltiples clientes

&nbsp;   

&nbsp;   Args:

&nbsp;       customers: Lista de clientes

&nbsp;       

&nbsp;   Returns:

&nbsp;       Lista de predicciones

&nbsp;   """

&nbsp;   try:

&nbsp;       results = \[]

&nbsp;       for customer in customers:

&nbsp;           pred = predict\_churn(customer)

&nbsp;           results.append(pred.dict())

&nbsp;       

&nbsp;       return {

&nbsp;           "total\_customers": len(customers),

&nbsp;           "predictions": results

&nbsp;       }

&nbsp;       

&nbsp;   except Exception as e:

&nbsp;       raise HTTPException(status\_code=500, detail=f"Error en batch: {str(e)}")





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   import uvicorn

&nbsp;   uvicorn.run(app, host="0.0.0.0", port=8000)

```



---



\## \*\*Paso 3: Ejecutar la API\*\*

```bash

\# Ejecutar en desarrollo

uvicorn src.api:app --reload --host 0.0.0.0 --port 8000



\# O con python directamente

python src/api.py

```



Acceder a:

\- API: `http://localhost:8000`

\- Documentación Swagger: `http://localhost:8000/docs`

\- Redoc: `http://localhost:8000/redoc`



---



\## \*\*Paso 4: Probar la API\*\*



\*\*Usando curl:\*\*

```bash

curl -X POST "http://localhost:8000/predict" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"tenure": 12, "monthly\_charges": 70.5, "total\_charges": 846.0}'

```



\*\*Usando Python:\*\*

```python

import requests



url = "http://localhost:8000/predict"

data = {

&nbsp;   "tenure": 12.0,

&nbsp;   "monthly\_charges": 70.5,

&nbsp;   "total\_charges": 846.0

}



response = requests.post(url, json=data)

print(response.json())

```



---



\## \*\*Paso 5: Despliegue en producción\*\*



\### \*\*Opción A: Servidor con systemd\*\*



Crear servicio en `/etc/systemd/system/telcovision.service`:

```ini

\[Unit]

Description=TelcoVision Churn Prediction API

After=network.target



\[Service]

User=www-data

WorkingDirectory=/opt/telcovision

Environment="PATH=/opt/telcovision/venv/bin"

ExecStart=/opt/telcovision/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000



\[Install]

WantedBy=multi-user.target

```



Activar:

```bash

sudo systemctl enable telcovision

sudo systemctl start telcovision

sudo systemctl status telcovision

```



---



\### \*\*Opción B: Docker\*\*



Crear `Dockerfile`:

```dockerfile

FROM python:3.9-slim



WORKDIR /app



COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt



COPY . .



EXPOSE 8000



CMD \["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

```



Construir y ejecutar:

```bash

docker build -t telcovision-api .

docker run -p 8000:8000 telcovision-api

```



---



\### \*\*Opción C: Servicios Cloud\*\*



\*\*AWS (Elastic Beanstalk):\*\*

```bash

eb init -p python-3.9 telcovision-api

eb create telcovision-env

eb deploy

```



\*\*Google Cloud (Cloud Run):\*\*

```bash

gcloud builds submit --tag gcr.io/PROJECT\_ID/telcovision

gcloud run deploy --image gcr.io/PROJECT\_ID/telcovision --platform managed

```



\*\*Azure (App Service):\*\*

```bash

az webapp up --name telcovision --runtime "PYTHON:3.9"

```



---



\# 🎨 Opción 2: Aplicación Web con Streamlit



\## \*\*¿Cuándo usar Streamlit?\*\*



\- ✅ Demos y prototipos rápidos

\- ✅ Dashboards internos

\- ✅ Aplicaciones para data scientists

\- ✅ No requiere conocimientos de frontend



---



\## \*\*Paso 1: Instalar Streamlit\*\*

```bash

pip install streamlit plotly

```



---



\## \*\*Paso 2: Crear app (`src/app\_streamlit.py`)\*\*

```python

"""

Aplicación Streamlit para predicción de churn - TelcoVision

"""



import streamlit as st

import joblib

import pandas as pd

import yaml

import plotly.graph\_objects as go



\# Configuración de página

st.set\_page\_config(

&nbsp;   page\_title="TelcoVision - Predicción de Churn",

&nbsp;   page\_icon="📊",

&nbsp;   layout="wide"

)



\# Cargar modelo

@st.cache\_resource

def load\_model():

&nbsp;   with open('params.yaml', 'r') as f:

&nbsp;       params = yaml.safe\_load(f)

&nbsp;   return joblib.load(params\['paths']\['model\_path'])



model = load\_model()



\# Título

st.title("📊 TelcoVision - Predicción de Churn")

st.markdown("Sistema de predicción de abandono de clientes en telecomunicaciones")



\# Sidebar con inputs

st.sidebar.header("Datos del Cliente")



\# Inputs (ajustar según tus features reales)

tenure = st.sidebar.number\_input("Meses de antigüedad", min\_value=0, max\_value=120, value=12)

monthly\_charges = st.sidebar.number\_input("Cargo mensual ($)", min\_value=0.0, max\_value=200.0, value=70.5)

total\_charges = st.sidebar.number\_input("Cargo total ($)", min\_value=0.0, max\_value=10000.0, value=846.0)



\# Botón de predicción

if st.sidebar.button("Predecir Churn", type="primary"):

&nbsp;   # Crear DataFrame

&nbsp;   data = pd.DataFrame({

&nbsp;       'tenure': \[tenure],

&nbsp;       'monthly\_charges': \[monthly\_charges],

&nbsp;       'total\_charges': \[total\_charges]

&nbsp;   })

&nbsp;   

&nbsp;   # Predecir

&nbsp;   churn\_proba = model.predict\_proba(data)\[0, 1]

&nbsp;   will\_churn = churn\_proba >= 0.5

&nbsp;   

&nbsp;   # Mostrar resultados

&nbsp;   col1, col2, col3 = st.columns(3)

&nbsp;   

&nbsp;   with col1:

&nbsp;       st.metric("Probabilidad de Churn", f"{churn\_proba\*100:.1f}%")

&nbsp;   

&nbsp;   with col2:

&nbsp;       st.metric("Predicción", "CHURN ❌" if will\_churn else "NO CHURN ✅")

&nbsp;   

&nbsp;   with col3:

&nbsp;       if churn\_proba < 0.3:

&nbsp;           risk = "Bajo 🟢"

&nbsp;       elif churn\_proba < 0.6:

&nbsp;           risk = "Medio 🟡"

&nbsp;       else:

&nbsp;           risk = "Alto 🔴"

&nbsp;       st.metric("Nivel de Riesgo", risk)

&nbsp;   

&nbsp;   # Gauge chart

&nbsp;   fig = go.Figure(go.Indicator(

&nbsp;       mode = "gauge+number",

&nbsp;       value = churn\_proba \* 100,

&nbsp;       domain = {'x': \[0, 1], 'y': \[0, 1]},

&nbsp;       title = {'text': "Probabilidad de Churn (%)"},

&nbsp;       gauge = {

&nbsp;           'axis': {'range': \[None, 100]},

&nbsp;           'bar': {'color': "darkred" if churn\_proba > 0.6 else "orange" if churn\_proba > 0.3 else "green"},

&nbsp;           'steps' : \[

&nbsp;               {'range': \[0, 30], 'color': "lightgreen"},

&nbsp;               {'range': \[30, 60], 'color': "lightyellow"},

&nbsp;               {'range': \[60, 100], 'color': "lightcoral"}

&nbsp;           ],

&nbsp;           'threshold': {

&nbsp;               'line': {'color': "red", 'width': 4},

&nbsp;               'thickness': 0.75,

&nbsp;               'value': 50

&nbsp;           }

&nbsp;       }

&nbsp;   ))

&nbsp;   

&nbsp;   st.plotly\_chart(fig, use\_container\_width=True)

&nbsp;   

&nbsp;   # Recomendaciones

&nbsp;   st.subheader("📋 Recomendaciones")

&nbsp;   if will\_churn:

&nbsp;       st.warning("""

&nbsp;       \*\*Acciones sugeridas:\*\*

&nbsp;       - Contactar al cliente proactivamente

&nbsp;       - Ofrecer descuentos o promociones personalizadas

&nbsp;       - Revisar experiencia del cliente y satisfacción

&nbsp;       - Evaluar calidad de servicio

&nbsp;       """)

&nbsp;   else:

&nbsp;       st.success("""

&nbsp;       \*\*Cliente con baja probabilidad de churn:\*\*

&nbsp;       - Mantener calidad de servicio actual

&nbsp;       - Considerar programas de fidelización

&nbsp;       - Monitorear satisfacción periódicamente

&nbsp;       """)

```



---



\## \*\*Paso 3: Ejecutar Streamlit\*\*

```bash

streamlit run src/app\_streamlit.py

```



Acceder a: `http://localhost:8501`



---



\## \*\*Paso 4: Desplegar Streamlit\*\*



\*\*Streamlit Cloud (gratis):\*\*

1\. Pushear código a GitHub

2\. Ir a \[share.streamlit.io](https://share.streamlit.io)

3\. Conectar repositorio

4\. Seleccionar `src/app\_streamlit.py`

5\. Deploy



---



\# 📦 Opción 3: Procesamiento por Lotes (Batch)



Para procesar grandes volúmenes de clientes de una vez.



\## \*\*Script de batch (`src/batch\_predict.py`)\*\*

```python

"""

Predicción batch para múltiples clientes

"""



import joblib

import pandas as pd

import yaml

from datetime import datetime



def batch\_predict(input\_file: str, output\_file: str):

&nbsp;   """

&nbsp;   Predecir churn para un archivo CSV de clientes

&nbsp;   

&nbsp;   Args:

&nbsp;       input\_file: CSV con datos de clientes

&nbsp;       output\_file: CSV de salida con predicciones

&nbsp;   """

&nbsp;   # Cargar modelo

&nbsp;   with open('params.yaml', 'r') as f:

&nbsp;       params = yaml.safe\_load(f)

&nbsp;   model = joblib.load(params\['paths']\['model\_path'])

&nbsp;   

&nbsp;   # Cargar datos

&nbsp;   df = pd.read\_csv(input\_file)

&nbsp;   print(f"Procesando {len(df)} clientes...")

&nbsp;   

&nbsp;   # Predecir

&nbsp;   df\['churn\_probability'] = model.predict\_proba(df)\[:, 1]

&nbsp;   df\['will\_churn'] = df\['churn\_probability'] >= 0.5

&nbsp;   df\['risk\_level'] = pd.cut(

&nbsp;       df\['churn\_probability'],

&nbsp;       bins=\[0, 0.3, 0.6, 1.0],

&nbsp;       labels=\['Bajo', 'Medio', 'Alto']

&nbsp;   )

&nbsp;   

&nbsp;   # Guardar

&nbsp;   df.to\_csv(output\_file, index=False)

&nbsp;   print(f"Resultados guardados en: {output\_file}")

&nbsp;   

&nbsp;   # Resumen

&nbsp;   print(f"\\nResumen:")

&nbsp;   print(f"- Total clientes: {len(df)}")

&nbsp;   print(f"- Churn predicho: {df\['will\_churn'].sum()} ({df\['will\_churn'].mean()\*100:.1f}%)")

&nbsp;   print(f"- Riesgo Alto: {(df\['risk\_level']=='Alto').sum()}")

&nbsp;   print(f"- Riesgo Medio: {(df\['risk\_level']=='Medio').sum()}")

&nbsp;   print(f"- Riesgo Bajo: {(df\['risk\_level']=='Bajo').sum()}")



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   batch\_predict(

&nbsp;       input\_file='data/clientes\_nuevos.csv',

&nbsp;       output\_file=f'predictions/batch\_{datetime.now().strftime("%Y%m%d\_%H%M%S")}.csv'

&nbsp;   )

```



---



\# 🐳 Opción 4: Despliegue con Docker Compose



Para desplegar API + Base de Datos + Monitoring.



\## \*\*`docker-compose.yml`\*\*

```yaml

version: '3.8'



services:

&nbsp; api:

&nbsp;   build: .

&nbsp;   ports:

&nbsp;     - "8000:8000"

&nbsp;   environment:

&nbsp;     - MODEL\_PATH=/app/models/model.joblib

&nbsp;   volumes:

&nbsp;     - ./models:/app/models

&nbsp;     - ./logs:/app/logs

&nbsp;   depends\_on:

&nbsp;     - db

&nbsp;   restart: unless-stopped



&nbsp; db:

&nbsp;   image: postgres:14

&nbsp;   environment:

&nbsp;     POSTGRES\_DB: telcovision

&nbsp;     POSTGRES\_USER: admin

&nbsp;     POSTGRES\_PASSWORD: secure\_password

&nbsp;   volumes:

&nbsp;     - postgres\_data:/var/lib/postgresql/data

&nbsp;   ports:

&nbsp;     - "5432:5432"



&nbsp; monitoring:

&nbsp;   image: prom/prometheus

&nbsp;   ports:

&nbsp;     - "9090:9090"

&nbsp;   volumes:

&nbsp;     - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml



volumes:

&nbsp; postgres\_data:

```



Ejecutar:

```bash

docker-compose up -d

```



---



\# 📊 Monitoreo y Mantenimiento



\## \*\*Métricas a Monitorear\*\*



1\. \*\*Performance del Modelo:\*\*

&nbsp;  - Accuracy, Precision, Recall en producción

&nbsp;  - Data drift (cambios en distribución de datos)

&nbsp;  - Prediction drift (cambios en distribución de predicciones)



2\. \*\*Performance del Sistema:\*\*

&nbsp;  - Latencia de respuesta (target: <100ms)

&nbsp;  - Throughput (requests/segundo)

&nbsp;  - Uso de memoria y CPU

&nbsp;  - Tasa de errores



3\. \*\*Negocio:\*\*

&nbsp;  - Tasa de churn real vs predicho

&nbsp;  - ROI de acciones preventivas

&nbsp;  - Costo de falsos positivos/negativos



\## \*\*Re-entrenamiento\*\*



Configurar pipeline automático:

```bash

\# Cron job mensual

0 0 1 \* \* cd /opt/telcovision \&\& dvc repro \&\& git add . \&\& git commit -m "Reentrenamiento mensual" \&\& dvc push

```



---



\# 🔒 Seguridad



\## \*\*Buenas Prácticas\*\*



1\. \*\*Autenticación:\*\*

```python

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials



security = HTTPBearer()



@app.post("/predict")

def predict(customer: CustomerData, credentials: HTTPAuthorizationCredentials = Depends(security)):

&nbsp;   # Validar token

&nbsp;   if not validate\_token(credentials.credentials):

&nbsp;       raise HTTPException(status\_code=401, detail="Invalid token")

&nbsp;   # ... predicción

```



2\. \*\*Rate Limiting:\*\*

```python

from slowapi import Limiter

from slowapi.util import get\_remote\_address



limiter = Limiter(key\_func=get\_remote\_address)

app.state.limiter = limiter



@app.post("/predict")

@limiter.limit("100/minute")

def predict(request: Request, customer: CustomerData):

&nbsp;   # ...

```



3\. \*\*HTTPS:\*\*

```bash

uvicorn src.api:app --host 0.0.0.0 --port 8000 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem

```



---



\# 📝 Checklist de Despliegue



\- \[ ] Modelo entrenado y validado

\- \[ ] API implementada y probada localmente

\- \[ ] Tests de integración pasando

\- \[ ] Documentación actualizada

\- \[ ] Variables de entorno configuradas

\- \[ ] Logging implementado

\- \[ ] Monitoreo configurado

\- \[ ] Backup automatizado

\- \[ ] Plan de rollback definido

\- \[ ] Equipo capacitado



---



\# 🎓 Recursos Adicionales



\- \[FastAPI Documentation](https://fastapi.tiangolo.com)

\- \[Streamlit Documentation](https://docs.streamlit.io)

\- \[Docker Documentation](https://docs.docker.com)

\- \[MLOps Best Practices](https://ml-ops.org)



---



\*\*Versión:\*\* 1.0  

\*\*Última actualización:\*\* Noviembre 2025  

\*\*Proyecto:\*\* TelcoVision

