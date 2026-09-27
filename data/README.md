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