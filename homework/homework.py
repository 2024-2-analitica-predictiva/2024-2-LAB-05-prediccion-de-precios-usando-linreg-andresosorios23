#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def main():

    train_data = pd.read_csv("files/input/train_data.csv.zip")
    test_data = pd.read_csv("files/input/test_data.csv.zip")

    # Paso 1
    train_data["Age"] = 2021 - train_data["Year"]
    test_data["Age"] = 2021 - test_data["Year"]

    train_data.drop(columns=["Year", "Car_Name"], inplace=True)
    test_data.drop(columns=["Year", "Car_Name"], inplace=True)

    # Paso 2
    x_train = train_data.drop(columns=["Present_Price"])
    y_train = train_data["Present_Price"]

    x_test = test_data.drop(columns=["Present_Price"])
    y_test = test_data["Present_Price"]

    # Paso 3

    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = x_train.columns.difference(categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("regression", LinearRegression(n_jobs=-1)),
        ]
    )

    # Paso 4
    parameters = {
        "feature_selection__k": range(1, x_train.shape[1] + 5),
        "regression__fit_intercept": [True, False],
        "regression__positive": [True, False],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=parameters,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(x_train, y_train)

    # Paso 5

    import gzip
    import pickle

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(grid_search, f)

    # Paso 6

    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, grid_search.predict(x_train)),
        "mse": mean_squared_error(y_train, grid_search.predict(x_train)),
        "mad": median_absolute_error(y_train, grid_search.predict(x_train)),
    }

    with open("files/output/metrics.json", "w") as f:
        json.dump(train_metrics, f)

    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, grid_search.predict(x_test)),
        "mse": mean_squared_error(y_test, grid_search.predict(x_test)),
        "mad": median_absolute_error(y_test, grid_search.predict(x_test)),
    }

    with open("files/output/metrics.json", "a") as f:
        f.write("\n")
        json.dump(test_metrics, f)


if __name__ == "__main__":
    main()
