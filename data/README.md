# Dataset

## Bank Marketing Dataset

Este proyecto utiliza el conjunto de datos **Bank Marketing** del
UCI Machine Learning Repository.

El problema corresponde a una clasificación binaria cuyo objetivo es
predecir si un cliente contratará un depósito a plazo.

La variable objetivo es:

- `y = yes`: el cliente contrató el depósito.
- `y = no`: el cliente no contrató el depósito.

## Obtención de los datos

Los archivos originales no se incluyen en el repositorio debido a que
los datos crudos se mantienen fuera del control de versiones.

Para reproducir el proyecto:

```bash
cd data

curl -L \
  "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip" \
  -o bank-marketing.zip

unzip bank-marketing.zip
unzip bank.zip

El archivo utilizado para el entrenamiento es:

data/bank-full.csv
Dimensiones del dataset

El archivo utilizado contiene:

45.211 observaciones.
16 variables predictoras originales.
1 variable objetivo (y).

Durante el modelamiento se excluyó la variable duration, por lo que
el modelo final utiliza 15 variables de entrada.

Valores desconocidos

El dataset no presenta valores nulos técnicos en las variables utilizadas.

Algunas variables categóricas contienen el valor unknown.
Este valor se conserva como una categoría explícita del dataset y no se
trata como un valor nulo.

Control de versiones

Los archivos:

*.csv
*.zip

se encuentran excluidos mediante .gitignore, por lo que los datos
crudos no se almacenan en el repositorio GitHub.