#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/26

import numpy as np
import pickle

from mnist import load_mnist
from three_layer_neural_network import forward


def get_test_data():
    (_, _), (test_img, test_label) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return test_img, test_label


def init_network():
    with open("db/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


if __name__ == "__main__":
    test_img, test_label = get_test_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(test_img)):
        y = forward(network, test_img[i])
        p = np.argmax(y)
        if p == test_label[i]:
            accuracy_cnt += 1

    print("Accuracy: %f" % (float(accuracy_cnt) / len(test_img)))
