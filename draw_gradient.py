#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/27

import numpy as np
import matplotlib.pylab as plt
from numerical_function import numerical_gradient


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def fun(x):
    return x[0]*x[1]


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
    
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.5)
    x1 = np.arange(-2, 2.5, 0.5)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()

    bind = np.array([X, Y]).reshape(-1, 2)
    
    grad = numerical_gradient(fun, bind)
    print(grad)
    grad = grad.reshape(2, -1)
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.show()

    