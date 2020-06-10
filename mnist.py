#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/25
"""
mnist 图像数据:
32bit Magic Number + 32bit number of images + 32bit row + 32bit column
+ 8bit pixels ...

mnist 标签数据:
32bit Magic Number + 32bit number of items
+ 8bit labels ...
"""

import numpy as np
import pickle
import os

IMAGE_SIZE = 784
mnist_path = {
    'train_img': "db/train-images-idx3-ubyte",
    'train_label': "db/train-labels-idx1-ubyte",
    'test_img': "db/test-images-idx3-ubyte",
    'test_label': "db/test-labels-idx1-ubyte"
}
mnist_pkl_path = "db/mnist.pkl"


def load_label(path):
    with open(path, 'rb') as label_file:
        labels = np.frombuffer(label_file.read(), np.uint8, offset=8)
    return labels


def load_img(path):
    with open(path, 'rb') as data_file:
        data = np.frombuffer(data_file.read(), np.uint8, offset=16)
    data = data.reshape(-1, IMAGE_SIZE)
    return data


def generate_dataset():
    dataset = {}
    dataset['train_img'] = load_img(mnist_path['train_img'])
    dataset['train_label'] = load_label(mnist_path['train_label'])
    dataset['test_img'] = load_img(mnist_path['test_img'])
    dataset['test_label'] = load_label(mnist_path['test_label'])
    return dataset


def convert_onehot_label(X):
    onehot_label = np.zeros((X.size, 10))
    for idx, row in enumerate(onehot_label):
        row[X[idx]] = 1
    return onehot_label


def dump_mnist():
    dataset = generate_dataset()
    with open(mnist_pkl_path, 'wb') as mnist_file:
        pickle.dump(dataset, mnist_file, -1)


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    :param normalize:
        True: 将图像像素正则化为[0.0, 1.0]
        False: 像素值为[0, 255]
    :param flatten:
        True: 将图像展开为以为数组
        False: 返回 1x28x28 数据
    :param one_hot_label:
        True: 标签作为 one-hot 数组返回， 如 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        False: 返回单个标签， 如 2
    :return: (训练图像，训练标签), (测试图像， 测试标签)
    """
    if not os.path.exists(mnist_pkl_path):
        dump_mnist()
    with open(mnist_pkl_path, 'rb') as mnist_pkl:
        dataset = pickle.load(mnist_pkl)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] = dataset[key] / 255.0

    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = convert_onehot_label(dataset[key])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == "__main__":
    dump_mnist()
