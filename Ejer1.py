# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:06:20 2024

@author: ddiaz
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

def entrenar(X, y, epocas, lr):
    np.random.seed(1)
    pesos = np.random.rand(X.shape[1], 1)
    bias = np.random.rand(1)

    for epoca in range(epocas):
        entrada = np.dot(X, pesos) + bias
        salida = funcion_escalon(entrada)

        error = y - salida

        # Debido a que no usamos una función de activación diferenciable, la actualización de pesos se basa en el error directo
        d_salida = error

        pesos += X.T.dot(d_salida) * lr
        bias += np.sum(d_salida) * lr

        if epoca % 100 == 0:
            print(f"Error en epoch {epoca}: {np.mean(np.abs(error))}")

    return pesos, bias

def predecir(X, pesos, bias):
    entrada = np.dot(X, pesos) + bias
    salida = funcion_escalon(entrada)
    return salida

iris = load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

epocas = 1000
lr = 0.4

pesos, bias = entrenar(X_train, y_train, epocas, lr)

salidas = predecir(X_test, pesos, bias)
predicciones = (salidas > 0.5).astype(int)
precision = np.mean(predicciones == y_test)
print(f"Precisión en los datos de prueba: {precision * 100}%")
print("Salidas predichas:")
print(predicciones)
print(f"Necesito de {epocas} épocas para entrenar")
