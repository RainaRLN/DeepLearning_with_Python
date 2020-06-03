#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/06/03

import numpy as np
from mnist import load_mnist
from bp_two_layer_net import TwoLayerNet


if __name__ == "__main__":
    (train_img, train_label), (_, _) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = train_img[:3]
    t_batch = train_label[:3]
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ": " + str(diff))
