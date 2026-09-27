from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


print("\n=== CARGANDO MODELO ===")

model = joblib.load(MODEL_PATH)

print("Modelo cargado correctamente.")


# Ejemplo de un cliente nuevo
new_customer = {
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
    "poutcome": "unknown",
}


df_new = pd.DataFrame([new_customer])


print("\n=== OBSERVACIÓN NUEVA ===")
print(df_new)


prediction = model.predict(df_new)[0]


classes = model.named_steps["classifier"].classes_

positive_class_index = list(classes).index("yes")

probability = model.predict_proba(df_new)[0][positive_class_index]


print("\n=== PREDICCIÓN ===")
print(f"Clase predicha: {prediction}")
print(f"Probabilidad de yes: {probability:.4f}")