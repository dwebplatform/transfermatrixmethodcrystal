import matplotlib.pyplot as plt
from layer import Layer
import numpy as np
from numpy import linalg as LA
from util import getRefractiveIndexForLamda, getReflection, getMatrixAbsDirect, getThreeCalcMatrix
import math
from read_f import readExperimentalData, readExperimentalDataFromThreeCols

CA_data = readExperimentalData('CA.txt')
PVK_data = readExperimentalData('PVK.txt')
Glass_data = readExperimentalDataFromThreeCols('Стекло.txt')


def getSuperMatrix(leftMatrix, glassLayer, rightMatrix):
    [leftMatrixAbsDirect, directDAbsMatrix, rightMatrixAbsDirect] = [getMatrixAbsDirect(
        leftMatrix), glassLayer.getDMatrix(), getMatrixAbsDirect(rightMatrix)]
    return getThreeCalcMatrix(leftMatrixAbsDirect, glassLayer.getDMatrix(), rightMatrixAbsDirect)


def getPlotData(resultMatrix, superResultMatrix):
    r = getReflection(resultMatrix)
    T = 1 / (superResultMatrix[0][0])
    R = superResultMatrix[1][0]/superResultMatrix[0][0]
    return [r, T, R]


def matrixForPeriod(layers):
    [M1Matrix, M2Matrix] = map(matrixForLayer, layers)
    MofTwoMatrix = np.dot(M1Matrix, M2Matrix)
    return MofTwoMatrix


def matrixForLayer(layer):
    result = np.dot(layer.getPMatrix(), layer.getDMatrix())
    result = np.dot(result, layer.getPReverseMatrix())
    return result


def getAirP1Matrix(angle, n):
    return np.array(
        [[1, 1], [n*math.cos(angle), -n*math.cos(angle)]])


def getGlassPSMatrix(angle, n):
    return np.array(
        [[1, 1], [n*math.cos(angle), -n*math.cos(angle)]])


# layers =[layerFirst, layerSecond]
def getResultMatrix(layers, d):
    [layerFirst, layerSecond] = layers
    AllMMatrix = LA.matrix_power(matrixForPeriod(layers), d)
    # result matrix
    resultMatrix = np.dot(LA.inv(getAirP1Matrix(
        layerFirst.alfa0, layerFirst.n0)), AllMMatrix)
    return resultMatrix


# ns = 1.52
# длина волны, продабуированная
lamdaList = list(range(280, 800, 1))
#
plotArray = []
# параметры начального луча
n0 = 1
alfa0 = (math.pi/180)*0

for lamda in lamdaList:
    # ns стекло, n1 PVK, n2 CA
    [ns, n1, n2] = [getRefractiveIndexForLamda(Glass_data, lamda, 1.52), getRefractiveIndexForLamda(
        PVK_data, lamda, 1.683), getRefractiveIndexForLamda(CA_data, lamda, 1.475)]
    # ширина слоя h1, h2, и число периодов
    [h1, h2, d] = [50, 150, 10]
    # первый и второй слои
    [layerFirst, layerSecond] = [Layer(lamda, n1, h1), Layer(lamda, n2, h2)]
    # матрица всех периодов
    resultMatrix = getResultMatrix([layerFirst, layerSecond], d)
    # последний слой стекло
    PSMatrix = getGlassPSMatrix(layerFirst.alfa0, ns)
    # вычисляем матрицу левую
    leftMatrix = np.dot(resultMatrix, PSMatrix)
    glassLayer = Layer(lamda, ns, 1000000)
    # правая матрица
    rightMatrix = np.dot(glassLayer.getPReverseMatrix(),
                         getAirP1Matrix(layerFirst.alfa0, layerFirst.n0))

    superResultMatrix = getSuperMatrix(leftMatrix, glassLayer, rightMatrix)
    # вычисляем коэффициент отражения
    [r, T, R] = getPlotData(resultMatrix, superResultMatrix)
    plotArray.append(R.real)

plt.plot(plotArray, color='black')
plt.show()
