# Bank Marketing Prediction API

El objetivo del proyecto es transformar un modelo de Machine Learning
entrenado localmente en un servicio de inferencia reproducible mediante
FastAPI.

El flujo implementado es:

```text
Dataset
   ↓
Exploración de datos
   ↓
Preparación
   ↓
Entrenamiento
   ↓
Pipeline de Machine Learning
   ↓
Serialización en model.pkl
   ↓
FastAPI
   ↓
API HTTP

```

## 1. Problema de Machine Learning

Se utiliza el dataset Bank Marketing del UCI Machine Learning
Repository.

El problema corresponde a una clasificación binaria supervisada cuyo
objetivo es predecir si un cliente contratará un depósito a plazo o no.

La variable objetivo y es:

yes: el cliente contrata el depósito.
no: el cliente no contrata el depósito.

El dataset utilizado contiene:

45.211 observaciones
16 variables predictoras originales
1 variable objetivo

La distribución aproximada del target es:

Clase	Porcentaje
no	88,30 %
yes	11,70 %

Por lo tanto, existe un desbalance importante entre las clases.

## 2. Variables utilizadas

El modelo final utiliza 15 variables predictoras.

Variables numéricas
age
balance
day
campaign
pdays
previous
Variables categóricas
job
marital
education
default
housing
loan
contact
month
poutcome

La variable:

duration

se excluyó como decisión de modelamiento para evitar incorporar
información que no se considera disponible en el momento definido para realizar la predicción (antes de realizar la llamada).

Las categorías unknown presentes en el dataset se conservan como
categorías explícitas y no son tratadas como valores nulos.

## 3. Exploración de datos

Durante la exploración se verificó:

número de filas y columnas
tipos de datos
presencia de valores nulos;
distribución de la variable objetivo
variables numéricas y categóricas
categorías disponibles
frecuencia de valores unknown.

El dataset no presentó valores nulos técnicos en las variables utilizadas.

La clase positiva yes representa aproximadamente el 11,70 % de las
observaciones, por lo que las métricas de evaluación deben interpretarse
considerando el desbalance de clases.

## 4. Pipeline de Machine Learning

El modelo se implementa utilizando un Pipeline de scikit-learn que
integra tanto el preprocesamiento como el estimador.

Datos originales
      ↓
ColumnTransformer
      │
      ├── Variables numéricas
      │       ↓
      │  StandardScaler
      │
      └── Variables categóricas
              ↓
         OneHotEncoder
              ↓
      LogisticRegression
Preprocesamiento numérico

Las variables numéricas son procesadas mediante:

StandardScaler()
Preprocesamiento categórico

Las variables categóricas son codificadas mediante:

OneHotEncoder(handle_unknown="ignore")
Modelo

El estimador utilizado es:

LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

El parámetro:

class_weight="balanced"

permite dar mayor peso relativo a la clase minoritaria durante el
entrenamiento.

## 5. Separación entrenamiento / test

La división de datos corresponde a:

80 % entrenamiento
20 % test

utilizando:

random_state=42
stratify=y

El uso de stratify=y permite mantener aproximadamente la misma
distribución de clases en los conjuntos de entrenamiento y test.

Resultados de la separación:

Train: 36.168 observaciones
Test:   9.043 observaciones

## 6. Evaluación del modelo

Resultados obtenidos sobre el conjunto de test:

Métrica	Resultado
Accuracy	0.7548
Precision	0.2662
Recall	0.6238
F1	0.3732
ROC-AUC	0.7722
Average Precision	0.4093
Matriz de confusión
[[6166, 1819],
 [ 398,  660]]

Interpretación:

TN = 6166
FP = 1819
FN = 398
TP = 660

Debido al desbalance de clases, el desempeño no se evalúa únicamente
mediante Accuracy también se consideran:
Recall
F1
ROC-AUC
Average Precision

El modelo alcanza un Recall de aproximadamente 62,38 % para la clase
positiva.

El uso de class_weight="balanced" favorece la detección de la clase
minoritaria, aunque aumenta el número de falsos positivos.

## 7. Serialización

El pipeline completo se serializa utilizando:

joblib

y se almacena en:

model/model.pkl

El archivo contiene en un único objeto:

Pipeline
├── ColumnTransformer
│   ├── StandardScaler
│   └── OneHotEncoder
│
└── LogisticRegression

Esto evita tener que reproducir manualmente el preprocesamiento durante
la inferencia.

También se genera:

model/metadata.json

que contiene información como:
versión del modelo
versión de Python
versión de scikit-learn
variables utilizadas
variables excluidas
métricas
tamaño de train y test
matriz de confusión.

## 8. Verificación del modelo serializado

El archivo:

verify_model.py

permite comprobar que model.pkl puede cargarse correctamente desde un
proceso diferente al utilizado durante el entrenamiento.

Ejecutar:

python verify_model.py

Ejemplo de resultado:

=== CARGANDO MODELO ===
Modelo cargado correctamente.

=== PREDICCIÓN ===
Clase predicha: no
Probabilidad de yes: 0.4635

##9. Estructura del repositorio
tarea-cloud-fastapi/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── README.md
│
├── docs/
│   ├── evidencia_local.png
│   └── evidencia_llamadas.txt
│
├── model/
│   ├── model.pkl
│   └── metadata.json
│
├── notebooks/
│
├── tests/
│   └── test_api.py
│
├── explore.py
├── train.py
├── verify_model.py
├── requirements.txt
├── runtime.txt
├── Procfile
├── .gitignore
└── README.md

## 10. Entorno de ejecución

El proyecto fue desarrollado y probado con:

Python 3.12.13

La versión se encuentra declarada en:

runtime.txt

Las principales dependencias se encuentran fijadas en:

requirements.txt

Incluyendo:

fastapi==0.115.0
uvicorn[standard]==0.30.6
scikit-learn==1.5.2
pandas==2.2.3
joblib==1.4.2
pydantic==2.9.2
pytest==8.3.3
httpx==0.27.2

## 11. Clonar el repositorio
git clone https://github.com/msagarra/tarea-cloud-msagarra.git

Entrar a la carpeta:

cd tarea-cloud-msagarra

## 12. Crear el entorno virtual

En macOS o Linux:

python3.12 -m venv .venv

Activar el entorno:

source .venv/bin/activate

Comprobar la versión de Python:

python --version

Resultado esperado:

Python 3.12.13

## 13. Instalar dependencias
pip install -r requirements.txt

## 14. Obtener el dataset

Los datos crudos no se almacenan en GitHub.

Las instrucciones completas se encuentran en:

data/README.md

Una vez descargado y extraído el dataset, el archivo utilizado para
entrenamiento debe encontrarse en:

data/bank-full.csv

## 15. Entrenar el modelo

Ejecutar:

python train.py

El entrenamiento genera:

model/model.pkl
model/metadata.json

## 16. Ejecutar la API

Levantar el servicio localmente mediante:

python -m uvicorn app.main:app --reload --port 8000

El servicio queda disponible en:

http://127.0.0.1:8000

## 17. Documentación Swagger

FastAPI genera automáticamente la documentación interactiva.

Abrir en el navegador:

http://127.0.0.1:8000/docs

Los endpoints disponibles son:

GET   /health
GET   /model-info
POST  /predict
POST  /predict-batch
GET   /docs

## 18. Endpoint /health

Permite verificar el estado del servicio y confirmar que el modelo y los
metadatos se encuentran cargados en memoria.

curl http://127.0.0.1:8000/health

Respuesta esperada:

{
  "status": "ok",
  "model_loaded": true,
  "metadata_loaded": true
}

## 19. Endpoint /model-info

Entrega información del modelo desplegado.

curl http://127.0.0.1:8000/model-info

Incluye:
versión del modelo
tipo de estimador
versión de Python
versión de scikit-learn
variables de entrada
variables excluidas
métricas de evaluación.

## 20. Endpoint /predict

Permite realizar una predicción individual.

Ejemplo:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'

Ejemplo de respuesta:

{
  "prediction": "no",
  "probability_yes": 0.4635,
  "prediction_probability": 0.5365,
  "model_version": "1.0.0"
}

La respuesta incluye además una marca temporal UTC.

## 21. Endpoint /predict-batch

Permite realizar predicciones sobre múltiples observaciones en una sola
petición.

El endpoint recibe una lista de objetos y devuelve una lista de
predicciones conservando el mismo orden de entrada.

En las pruebas locales se obtuvieron, para dos observaciones:

Observación 1
prediction = no
probability_yes = 0.4635

Observación 2
prediction = yes
probability_yes = 0.8912

## 22. Validación con Pydantic

Los datos de entrada se validan utilizando modelos Pydantic definidos
en:

app/schemas.py

Se validan:

tipos de datos;
campos obligatorios;
rangos numéricos;
categorías permitidas;
campos adicionales.

Por ejemplo:

{
  "age": 150,
  "job": "astronaut"
}

es una entrada inválida.

La API responde:

HTTP 422 Unprocessable Entity

antes de enviar esos datos al modelo.

## 23. Manejo de errores

Las entradas inválidas son rechazadas automáticamente mediante Pydantic
con código:

HTTP 422

Los errores internos producidos durante la inferencia son controlados y
devuelven:

HTTP 500

sin exponer trazas internas al cliente.

## 24. Pruebas automatizadas

Las pruebas se encuentran en:

tests/test_api.py

Se utiliza:

FastAPI TestClient
pytest

Para ejecutarlas:

python -m pytest -v

Las pruebas implementadas verifican:

/health;
/predict;
/predict-batch;
entrada inválida con respuesta HTTP 422.

Resultado obtenido:

4 passed

En una de las ejecuciones locales registradas:

4 passed, 25 warnings in 0.95s

Las advertencias corresponden a DeprecationWarning provenientes de
dependencias externas y no provocan fallos en las pruebas.

## 25. Evidencias

La evidencia de funcionamiento local se encuentra almacenada en:

docs/
Documentación Swagger
docs/evidencia_local.png
Llamadas HTTP
docs/evidencia_llamadas.txt

Este archivo contiene evidencia de:

predicción individual con HTTP 200 OK;
predicción por lote con HTTP 200 OK;
entrada inválida con HTTP 422 Unprocessable Entity.

## 26. Procfile

El comando declarado para iniciar el servicio en una plataforma compatible
es:

web: uvicorn app.main:app --host 0.0.0.0 --port $PORT

El puerto se obtiene desde la variable de entorno $PORT.

## 27. Reproducibilidad

Para reproducir el proyecto desde cero:

1. Clonar el repositorio
2. Crear el entorno virtual
3. Activar el entorno
4. Instalar requirements.txt
5. Descargar el dataset
6. Ejecutar train.py
7. Verificar model.pkl
8. Levantar FastAPI
9. Probar los endpoints
10. Ejecutar pytest

El repositorio no versiona:

entorno virtual .venv;
cachés de Python;
caché de pytest;
archivos .csv;
archivos .zip;
archivos .env;
credenciales o secretos.

## 28. Estado del despliegue

El servicio se encuentra validado y probado en localhost.

El despliegue público en una plataforma Cloud corresponde a una etapa
opcional adicional del proyecto.

## 29. Repositorio

Repositorio GitHub:

https://github.com/msagarra/tarea-cloud-msagarra

## 30. Autor

Matías Sagarra Barbano

Curso: Cloud Computing
Universidad Adolfo Ibáñez
Profesor: Ahmad Armoush