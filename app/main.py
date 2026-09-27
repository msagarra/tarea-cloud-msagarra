from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from app.schemas import CustomerObservation, PredictionResponse


# ============================================================
# 1. RUTAS DE LOS ARTEFACTOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

# ============================================================
# 2. ARTEFACTOS CARGADOS EN MEMORIA
# ============================================================

ARTIFACTS = {}

# ============================================================
# 3. CICLO DE VIDA DE LA APLICACIÓN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        ARTIFACTS["model"] = joblib.load(MODEL_PATH)

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as metadata_file:
            ARTIFACTS["metadata"] = json.load(metadata_file)

    except Exception as exc:
        raise RuntimeError(
            "No fue posible cargar los artefactos del modelo."
        ) from exc

    yield

    ARTIFACTS.clear()

    # ============================================================
# 4. CREAR LA APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="Bank Marketing Prediction API",
    description=(
        "API de inferencia para predecir si un cliente "
        "contratará un depósito a plazo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
# ============================================================
# 5. FUNCIÓN INTERNA DE PREDICCIÓN
# ============================================================

def make_prediction(observation: CustomerObservation):

    model = ARTIFACTS["model"]
    metadata = ARTIFACTS["metadata"]

    # Convertir el objeto Pydantic a diccionario y luego a DataFrame
    input_df = pd.DataFrame(
        [observation.model_dump()]
    )

    # Clase predicha
    prediction = model.predict(input_df)[0]

    # Probabilidades de todas las clases
    probabilities = model.predict_proba(input_df)[0]

    # Orden de clases aprendido por el modelo
    classes = model.named_steps[
        "classifier"
    ].classes_

    # Posición de la clase positiva "yes"
    positive_class_index = list(
        classes
    ).index("yes")

    probability_yes = probabilities[
        positive_class_index
    ]

    # Posición de la clase finalmente predicha
    predicted_class_index = list(
        classes
    ).index(prediction)

    prediction_probability = probabilities[
        predicted_class_index
    ]

    return {
        "prediction": str(prediction),
        "probability_yes": round(
            float(probability_yes),
            4,
        ),
        "prediction_probability": round(
            float(prediction_probability),
            4,
        ),
        "model_version": metadata[
            "model_version"
        ],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": "model" in ARTIFACTS,
        "metadata_loaded": "metadata" in ARTIFACTS,
    }

# ============================================================
# 6. INFORMACIÓN DEL MODELO
# ============================================================

@app.get("/model-info")
def model_info():

    metadata = ARTIFACTS["metadata"]

    return {
        "model_version": metadata["model_version"],
        "estimator": metadata["estimator"],
        "python_version": metadata["python_version"],
        "sklearn_version": metadata["sklearn_version"],
        "features": metadata["features"],
        "excluded_features": metadata["excluded_features"],
        "metrics": metadata["metrics"],
    }

# ============================================================
# 7. PREDICCIÓN INDIVIDUAL
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    observation: CustomerObservation,
):

    try:
        return make_prediction(observation)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Error al generar la predicción.",
        ) from exc

    # ============================================================
# 8. PREDICCIÓN POR LOTE
# ============================================================

@app.post(
    "/predict-batch",
    response_model=list[PredictionResponse],
)
def predict_batch(
    observations: list[CustomerObservation],
):

    try:
        return [
            make_prediction(observation)
            for observation in observations
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Error al generar las predicciones "
                "por lote."
            ),
        ) from exc