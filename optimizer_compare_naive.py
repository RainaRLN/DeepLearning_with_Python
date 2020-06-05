#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/06/04

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

from optimizer import SDG, Momentum, AdaGrad, Adam


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0 * y


if __name__ == "__main__":
    init_pos = (-7.0, 2.0)
    params = {}
    grads = {}
    optimizers = OrderedDict()
    optimizers['SDG'] = SDG(lr=0.95)
    optimizers['Momentum'] = Momentum(lr=0.1)
    optimizers['AdaGrad'] = AdaGrad(lr=1.5)
    optimizers['Adam'] = Adam(lr=0.3)

    fig = plt.figure()

    idx = 1

    for key in optimizers.keys():
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'],  params['y'] = init_pos[0], init_pos[1]
        grads['x'], grads['y'] = 0, 0

        iters_times = 30
        for i in range(iters_times):
            x_history.append(params['x'])
            y_history.append(params['y'])

            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)

        x = np.arange(-8, 8, 0.1)
        y = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        # x_history = np.array(x_history)
        # y_history = np.array(y_history)
        # z_history = f(x_history, y_history)
        # ax = fig.add_subplot(2, 2, idx, projection='3d')
        # ax.plot(x_history, y_history, z_history, 'ro-', linewidth=2)
        # ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

        sub_plt = fig.add_subplot(2, 2, idx)
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")
        
        idx += 1

    plt.show()
