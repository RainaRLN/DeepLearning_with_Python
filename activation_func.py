#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/19

import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    """
    Step function
    :param x: numpy.array
    :return: numpy.array
    """
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    """
    Sigmoid function
    :param x: numpy.array
    :return: numpy.array
    """
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """
    ReLU function
    :param x: numpy.array
    :return: numpy.array
    """
    return np.maximum(0, x)


def identity_function(x):
    """
    恒等函数 sigma function
    :param x: numpy.array
    :return: x
    """
    return x


# def softmax(x):
#     max_x = np.max(x)
#     exp_x = np.exp(x - max_x)
#     sum_exp_x = np.sum(exp_x)
#     y = exp_x / sum_exp_x
#     return y

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)

    fig = plt.figure()
    sub_pic1 = fig.add_subplot(2, 1, 1)
    sub_pic2 = fig.add_subplot(2, 1, 2)
    sub_pic1.plot(x, y1, label="step")
    sub_pic1.plot(x, y2, label="sigmoid")
    sub_pic2.plot(x, y3, label="ReLU")
    sub_pic1.legend()
    fig.show()
