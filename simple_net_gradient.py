#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/27

import numpy as np
from loss_function import cross_entropy_error
from activation_func import softmax
from numerical_function import numerical_gradient_no_batch


class simpleNet:
    def __init__(self, x, t):
        self.W = np.random.randn(2, 3)
        self.x = x
        self.t = t

    def predict(self):
        return np.dot(self.x, self.W)

    def loss(self):
        z = self.predict()
        y = softmax(z)
        loss = cross_entropy_error(y, self.t)
        return loss

    def f(self, w):
        self.W = w
        return self.loss()

    def set_x(self, x):
        self.x = x

    def set_w(self, w):
        self.W = w


if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    net = simpleNet(x, t)
    print(net.W)
    print(net.loss())
    print(numerical_gradient_no_batch(net.f, net.W))
