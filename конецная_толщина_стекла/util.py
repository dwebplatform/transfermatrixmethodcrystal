import numpy as np


def getThreeCalcMatrix(a, b, c):
    res = np.dot(a, b)
    res = np.dot(res, c)
    return res


def getMatrixAbsDirect(matrix):
    matrix[0][0] = matrix[0][0] * matrix[0][0].conjugate()
    matrix[0][1] = matrix[0][1] * matrix[0][1].conjugate()
    matrix[1][0] = matrix[1][0] * matrix[1][0].conjugate()
    matrix[1][1] = matrix[1][1] * matrix[1][1].conjugate()
    return matrix


def getMatrixForLayer(pMatrix, dMatrix, pMatrixReveresed):
    res = np.dot(pMatrix, dMatrix)
    res = np.dot(res, pMatrixReveresed)
    return res


def getReflection(matrix):
    A = matrix[0][0]
    C = matrix[1][0]
    amplitude = C / A
    r = amplitude * amplitude.conjugate()
    return r
