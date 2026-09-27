# Bank Marketing Prediction API

Proyecto desarrollado para la asignatura **Cloud Computing** del
Diploma/Magíster en Data Science de la Universidad Adolfo Ibáñez.

El objetivo es transformar un modelo de Machine Learning entrenado
localmente en un servicio de inferencia mediante **FastAPI**.

El proyecto cubre el flujo:

```text
Dataset
   ↓
Exploración de datos
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