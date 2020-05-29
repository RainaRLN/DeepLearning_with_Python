#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/18
import numpy as np
import abc


class PerceptronFactory:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def create_gate_perceptron(self, mode):
        perceptron = None
        if mode == "AND":
            perceptron = ANDGate(self.x1, self.x2)
        elif mode == "OR":
            perceptron = ORGate(self.x1, self.x2)
        elif mode == "NAND":
            perceptron = NANDGate(self.x1, self.x2)
        elif mode == "XOR":
            perceptron = XORGate(self.x1, self.x2)
        return perceptron


class Perceptron:
    def __init__(self, x1, x2):
        self.x = np.array([x1, x2])
        self.w = None
        self.b = None
        self.rst = None

    @abc.abstractmethod
    def __generate(self):
        pass

    def get_result(self):
        self.rst = np.sum(self.w * self.x) + self.b
        if self.rst <= 0:
            return 0
        else:
            return 1


class ANDGate(Perceptron):
    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        self.__generate()

    def __generate(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.7


class NANDGate(Perceptron):
    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        self.__generate()

    def __generate(self):
        self.w = np.array([-0.5, -0.5])
        self.b = 0.7


class ORGate(Perceptron):
    def __init__(self, x1, x2):
        super().__init__(x1, x2)
        self.__generate()

    def __generate(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.3


class XORGate:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def get_result(self):
        s1 = NANDGate(self.x1, self.x2).get_result()
        s2 = ORGate(self.x1, self.x2).get_result()
        y = ANDGate(s1, s2).get_result()
        return y


if __name__ == '__main__':
    gate = PerceptronFactory(1, 0).create_gate_perceptron("XOR")
    print(gate.get_result())
