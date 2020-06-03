#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/26

import numpy as np
import pickle

from mnist import load_mnist
from three_layer_neural_network import forward

BATCH_SIZE = 100


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
    for i in range(0, len(test_img), BATCH_SIZE):
        test_img_batch = test_img[i : i+BATCH_SIZE]
        y_batch = forward(network, test_img_batch)
        p = np.argmax(y_batch, axis=1)  # 行方向
        test_label_batch = test_label[i : i+BATCH_SIZE]
        accuracy_cnt += np.sum(test_label_batch == p)
    
    print("Accuracy: %f" % (float(accuracy_cnt) / len(test_img)) )
