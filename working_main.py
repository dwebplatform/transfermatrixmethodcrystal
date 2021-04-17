import matplotlib.pyplot as plt
from layer import Layer
import numpy as np
from numpy import linalg as LA
from util import getMatrixForLayer, getReflection
import math
from read_f import readExperimentalData

CA_data = readExperimentalData('CA.txt')
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
    M1Matrix = getMatrixForLayer(layerFirst.getPMatrix(
    ),  layerFirst.getDMatrix(), layerFirst.getPReverseMatrix())
    # if lamda in CA_data:

    n2 = 1.475
    h2 = 150

    # последний слой стекло
    PSMatrix = np.array(
        [[1, 1], [ns*math.cos(layerFirst.alfa0), -ns*math.cos(layerFirst.alfa0)]])
    layerSecond = Layer(lamda, n2, h2)
# матрица второго слоя

    M2Matrix = getMatrixForLayer(
        layerSecond.getPMatrix(), layerSecond.getDMatrix(), layerSecond.getPReverseMatrix())
#  матрица двух слоев
    MofTwoMatrix = np.dot(M1Matrix, M2Matrix)
    d = 10
# матрица всех периодов
    AllMMatrix = LA.matrix_power(MofTwoMatrix, d)
    resultMatrix = np.dot(LA.inv(P1Martix), AllMMatrix)

    # вычисляем матрицу финальную
    resultMatrix = np.dot(resultMatrix, PSMatrix)
    # вычисляем коэффициент отражения
    r = getReflection(resultMatrix)
    plotArray.append(r.real)
    # print(AllMMatrix)

plt.plot(plotArray, color='black')
plt.show()
