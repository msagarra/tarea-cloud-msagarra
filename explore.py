import pandas as pd


DATA_PATH = "data/bank-full.csv"


# 1. Cargar el dataset
df = pd.read_csv(DATA_PATH, sep=";")


# 2. Dimensiones
print("\n=== DIMENSIONES ===")
print(df.shape)


# 3. Primeras filas
print("\n=== PRIMERAS 5 FILAS ===")
print(df.head())


# 4. Nombres de las columnas
print("\n=== COLUMNAS ===")
print(df.columns.tolist())


# 5. Tipos de datos
print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)


# 6. Valores nulos
print("\n=== VALORES NULOS ===")
print(df.isnull().sum())


# 7. Distribución del target
print("\n=== DISTRIBUCIÓN DEL TARGET ===")
print(df["y"].value_counts())


print("\n=== DISTRIBUCIÓN PORCENTUAL DEL TARGET ===")
print(df["y"].value_counts(normalize=True) * 100)


# 8. Variables categóricas
categorical_cols = df.select_dtypes(include="object").columns.tolist()

print("\n=== VARIABLES CATEGÓRICAS ===")
print(categorical_cols)


# 9. Variables numéricas
numeric_cols = df.select_dtypes(exclude="object").columns.tolist()

print("\n=== VARIABLES NUMÉRICAS ===")
print(numeric_cols)

# 10. Cantidad de categorías por variable categórica
print("\n=== CANTIDAD DE CATEGORÍAS ===")

for col in categorical_cols:
    print(f"{col}: {df[col].nunique()}")


# 11. Valores de cada variable categórica
print("\n=== VALORES DE VARIABLES CATEGÓRICAS ===")

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())


# 12. Cantidad de valores 'unknown'
print("\n=== VALORES UNKNOWN ===")

for col in categorical_cols:
    unknown_count = (df[col] == "unknown").sum()

    if unknown_count > 0:
        porcentaje = unknown_count / len(df) * 100
        print(
            f"{col}: {unknown_count} "
            f"({porcentaje:.2f}%)"
        )