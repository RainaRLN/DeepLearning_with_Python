#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/26

import numpy as np


def numerical_diff(f, x):
    """
    :param: f: 函数f(x)
    :param x: x
    return: x点处的导数
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient_no_batch(f, x):
    """
    :param f:
    :param x:
    return: 返回函数f(*)在点x处的梯度
    """
    x = x.astype(float)
    h = 1e-4
    grad = np.zeros_like(x)  # 生成与x形状相同的全0数组

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_no_batch(f, x)

        return grad


def func(x):
    """
    f(x) = x^2
    """
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def gradient_descent(f, init_x, rate=0.01, step_num=100):
    """
    梯度下降法
    :param f: 函数
    :param init_x: 初始值
    :param rate: 学习率
    :param step_num: 下降次数
    return: 返回极小值
    """
    if init_x.dtype != 'float64':
        init_x = init_x.astype(float)
    x = init_x

    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        print(grad)
        x -= rate * grad

    return x


if __name__ == "__main__":
    init_x = np.array([6])
    x_min = gradient_descent(func, init_x, rate=0.3, step_num=10)
    print("%.6f" % x_min)
