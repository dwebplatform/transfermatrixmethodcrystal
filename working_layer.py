import math
import numpy as np
from numpy import linalg as LA

# n0*sin(alfa0) = ni *sin(alfai)


class Layer:
    n0 = 1
    alfa0 = (math.pi/180)*50

    def __init__(self, lamda, n, h):
        self.n = n
        self.h = h
        self.k = (2*math.pi)/lamda
        self.lamda = lamda

    def getRefractiveIndex(self):
        return self.n

    def getAngle(self):
        angleSin = (math.sin(self.alfa0) * self.n0) / self.n
        return math.asin(angleSin)

    def getK(self):
        return self.k * self.n * math.cos(self.getAngle())

    def getPMatrix(self):
        layerAngle = self.getAngle()
        return np.array([[1, 1], [self.n * math.cos(layerAngle), -self.n * math.cos(layerAngle)]])

    def getDMatrix(self):
        return np.array([[np.exp(1j * self.getK() * self.h), 0], [0, np.exp(-1j * self.getK() * self.h)]])

    def getPReverseMatrix(self):
        return LA.inv(self.getPMatrix())
