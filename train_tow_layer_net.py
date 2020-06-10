#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/28

import numpy as np
import matplotlib.pyplot as plt
import json

from loss_function import cross_entropy_error
from activation_func import softmax, sigmoid, sigmoid_grad
from numerical_function import numerical_gradient_no_batch
from mnist import load_mnist


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        :param input_size: 输入层神经元数
        :param hidden_size: 隐藏层神经元数
        :param output_size: 输出层神经元数
        :param weight_init_std: 权重标准差
        return: None
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.x = None
        self.t = None

    def set_x(self, x):
        self.x = x

    def set_t(self, t):
        self.t = t

    def predict(self):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(self.x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self):
        y = self.predict()
        return cross_entropy_error(y, self.t)

    def accuracy(self):
        y = self.predict()
        y = np.argmax(y, axis=1)
        t = np.argmax(self.t, axis=1)
        accuracy = np.sum(y == t) / float(self.x.shape[0])
        return accuracy

    def numerical_gradient(self):
        loss_W = lambda W: self.loss()
        grads = {}
        grads['W1'] = numerical_gradient_no_batch(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_no_batch(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_no_batch(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_no_batch(loss_W, self.params['b2'])
        return grads

    def gradient(self):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = self.x.shape[0]

        # forward
        a1 = np.dot(self.x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - self.t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(self.x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


if __name__ == "__main__":
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_times = 10000
    train_size = train_img.shape[0]
    batch_size = 100
    learn_rate = 0.1
    epoch = max(int(train_size / batch_size), 1)

    train_acc_list = []
    print("Start trainning ...")
    for i in range(iters_times):
        batch_mask = np.random.choice(train_size, batch_size)
        train_img_batch = train_img[batch_mask]
        train_label_batch = train_label[batch_mask]
        network.set_x(train_img_batch)
        network.set_t(train_label_batch)

        grad = network.gradient()

        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learn_rate * grad[key]

        if i % epoch == 0 or i == iters_times - 1:
            acc = network.accuracy()
            print(i, acc)
            train_acc_list.append(acc)

    params = {}
    params['W1'] = network.params['W1'].tolist()
    params['b1'] = network.params['b1'].tolist()
    params['W2'] = network.params['W2'].tolist()
    params['b2'] = network.params['b2'].tolist()

    f = open("./db/param_result/784x50x10-%.5f.json" % acc, 'w')
    f.write(json.dumps(params))
    f.close()
    plt.plot(np.arange(len(train_acc_list)), train_acc_list)
    plt.show()
