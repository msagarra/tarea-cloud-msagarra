from pathlib import Path
import json
import platform

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "y"

NUM_COLS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CAT_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

DROP_COLS = ["duration"]


# ============================================================
# 2. RUTAS PORTABLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# 3. CARGAR LOS DATOS
# ============================================================

print("\n=== CARGANDO DATASET ===")

df = pd.read_csv(DATA_PATH, sep=";")

print(f"Filas: {df.shape[0]}")
print(f"Columnas originales: {df.shape[1]}")


# ============================================================
# 4. SEPARAR FEATURES Y TARGET
# ============================================================

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]

print("\n=== FEATURES UTILIZADAS ===")
print(X.columns.tolist())

print(f"\nNúmero de features: {X.shape[1]}")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n=== TRAIN / TEST SPLIT ===")
print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")

print("\nDistribución target en train:")
print(y_train.value_counts(normalize=True))

print("\nDistribución target en test:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 6. PREPROCESAMIENTO
# ============================================================

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ]
)


# ============================================================
# 7. MODELO
# ============================================================

classifier = LogisticRegression(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=1000,
)


# ============================================================
# 8. PIPELINE COMPLETO
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# ============================================================
# 9. ENTRENAMIENTO
# ============================================================

print("\n=== ENTRENANDO MODELO ===")

pipeline.fit(X_train, y_train)

print("Entrenamiento completado.")


# ============================================================
# 10. PREDICCIONES
# ============================================================

y_pred = pipeline.predict(X_test)

classes = pipeline.named_steps["classifier"].classes_

positive_class_index = list(classes).index("yes")

y_proba = pipeline.predict_proba(X_test)[:, positive_class_index]


# ============================================================
# 11. MÉTRICAS
# ============================================================

y_test_binary = (y_test == "yes").astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test,
        y_pred,
        pos_label="yes",
    ),
    "recall": recall_score(
        y_test,
        y_pred,
        pos_label="yes",
    ),
    "f1": f1_score(
        y_test,
        y_pred,
        pos_label="yes",
    ),
    "roc_auc": roc_auc_score(
        y_test_binary,
        y_proba,
    ),
    "average_precision": average_precision_score(
        y_test_binary,
        y_proba,
    ),
}


print("\n=== MÉTRICAS ===")

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


print("\n=== REPORTE DE CLASIFICACIÓN ===")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)


# ============================================================
# 12. MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["no", "yes"],
)

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(cm)


# ============================================================
# 13. SERIALIZACIÓN DEL PIPELINE
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    pipeline,
    MODEL_PATH,
)

print(f"\nModelo guardado en: {MODEL_PATH}")


# ============================================================
# 14. METADATA
# ============================================================

metadata = {
    "model_version": "1.0.0",
    "estimator": "LogisticRegression",
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "target": TARGET,
    "positive_class": "yes",
    "features": list(X.columns),
    "numeric_features": NUM_COLS,
    "categorical_features": CAT_COLS,
    "excluded_features": DROP_COLS,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "metrics": {
        name: round(value, 4)
        for name, value in metrics.items()
    },
    "confusion_matrix": {
        "labels": ["no", "yes"],
        "values": cm.tolist(),
    },
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(
        metadata,
        metadata_file,
        indent=2,
        ensure_ascii=False,
    )


print(f"Metadata guardada en: {METADATA_PATH}")