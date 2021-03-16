import matplotlib.pyplot as plt
from layer import Layer
import numpy as np
from numpy import linalg as LA

import math
n0 = 1  # воздуха
ns = 1.52
lamdaList = list(range(280, 800, 1))
plotArray = []
for lamda in lamdaList:
    n1 = 1.683
    h1 = 50

    layerFirst = Layer(lamda, n1, h1)
    # первый слой воздух
    P1Martix = np.array([[1, 1],
                         [n0*math.cos(layerFirst.alfa0), -n0*math.cos(layerFirst.alfa0)]])


#  матрица первого слоя
#  first second third
    firstMatrix = layerFirst.getPMatrix()
    secondMatrix = layerFirst.getDMatrix()
    M1Matrix = np.dot(firstMatrix, secondMatrix)
    M1Matrix = np.dot(M1Matrix, layerFirst.getPReverseMatrix())

    n2 = 1.475
    h2 = 150
    # последний слой стекло
    PSMatrix = np.array(
        [[1, 1], [ns*math.cos(layerFirst.alfa0), -ns*math.cos(layerFirst.alfa0)]])
    layerSecond = Layer(lamda, n2, h2)
# матрица второго слоя

    M2Matrix = np.dot(layerSecond.getPMatrix(), layerSecond.getDMatrix())
    M2Matrix = np.dot(M2Matrix, layerSecond.getPReverseMatrix())

#  матрица двух слоев

    MofTwoMatrix = np.dot(M1Matrix, M2Matrix)

    d = 10
# матрица всех периодов
    AllMMatrix = LA.matrix_power(MofTwoMatrix, d)
    resultMatrix = np.dot(LA.inv(P1Martix), AllMMatrix)

    # вычисляем матрицу финальную
    resultMatrix = np.dot(resultMatrix, PSMatrix)

    A = resultMatrix[0][0]
    C = resultMatrix[1][0]
    # plotArray.append(lamda)
    amplitude = C / A
    r = amplitude * amplitude.conjugate()
    if (lamda == 280):
        print(r)
    plotArray.append(r.real)
    # print(AllMMatrix)

plt.plot(plotArray, 'o', color='black')
plt.show()
