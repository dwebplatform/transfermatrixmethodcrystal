import matplotlib.pyplot as plt
from layer import Layer
import numpy as np
from numpy import linalg as LA
from util import getMatrixForLayer, getReflection, getMatrixAbsDirect, getThreeCalcMatrix
import math
from read_f import readExperimentalData, readExperimentalDataFromThreeCols

CA_data = readExperimentalData('CA.txt')
PVK_data = readExperimentalData('PVK.txt')
Glass_data = readExperimentalDataFromThreeCols('Стекло.txt')


def getRefractiveIndexForLamda(experimentalData, lamda, defalut):
    n = defalut
    if lamda in experimentalData:
        n = complex(experimentalData[lamda])
    return n


n0 = 1  # воздуха
# ns = 1.52
lamdaList = list(range(280, 800, 1))
plotArray = []
for lamda in lamdaList:

    # стекло
    ns = getRefractiveIndexForLamda(Glass_data, lamda, 1.52)
    # PVK
    n1 = getRefractiveIndexForLamda(PVK_data, lamda, 1.683)
    h1 = 50

    layerFirst = Layer(lamda, n1, h1)
    # первый слой воздух
    P1Martix = np.array([[1, 1],
                         [n0*math.cos(layerFirst.alfa0), -n0*math.cos(layerFirst.alfa0)]])
#  матрица первого слоя
    M1Matrix = getMatrixForLayer(layerFirst.getPMatrix(
    ),  layerFirst.getDMatrix(), layerFirst.getPReverseMatrix())
    # if lamda in CA_data:

    n2 = getRefractiveIndexForLamda(CA_data, lamda, 1.475)
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

    leftMatrix = resultMatrix
    leftMatrixAbsDirect = getMatrixAbsDirect(leftMatrix)

    glassLayer = Layer(lamda, ns, 1000000)

    glassDMatrix = glassLayer.getDMatrix()
    directDAbsMatrix = getMatrixAbsDirect(glassDMatrix)
    # правая матрица
    rightMatrix = np.dot(glassLayer.getPReverseMatrix(), P1Martix)
    rightMatrixAbsDirect = getMatrixAbsDirect(rightMatrix)

    superResultMatrix = getThreeCalcMatrix(
        leftMatrixAbsDirect, glassDMatrix, rightMatrixAbsDirect)

    # вычисляем коэффициент отражения
    r = getReflection(resultMatrix)
    T = 1 / (superResultMatrix[0][0])
    R = superResultMatrix[1][0]/superResultMatrix[0][0]
    plotArray.append(R.real)
    # print(AllMMatrix)

plt.plot(plotArray, color='black')
plt.show()
