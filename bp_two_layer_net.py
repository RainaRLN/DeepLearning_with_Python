#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/06/02

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import json

from layers import Affine, Relu, SoftmaxWithLoss
from numerical_function import numerical_gradient
from mnist import load_mnist


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # f = open("./db/param_result/784x50x10-0.99162.json", 'r')
        # self.params = json.load(f)
        # f.close()
        # for key in ('W1', 'b1', 'W2', 'b2'):
        #     self.params[key] = np.array(self.params[key])

        # 创建各层的对象
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers_list = list(self.layers.values())
        layers_list.reverse()
        for layer in layers_list:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


if __name__ == "__main__":
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_times = 10000
    train_size = train_img.shape[0]
    batch_size = 100
    learn_rate = 0.1
    epoch = max(int(train_size / batch_size), 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    print("Start trainning ...")
    for i in range(iters_times):
        batch_mask = np.random.choice(train_size, batch_size)
        train_img_batch = train_img[batch_mask]
        train_label_batch = train_label[batch_mask]

        grad = network.gradient(train_img_batch, train_label_batch)

        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learn_rate * grad[key]

        if i % epoch == 0 or i == iters_times - 1:
            train_acc = network.accuracy(train_img_batch, train_label_batch)
            test_acc = network.accuracy(test_img, test_label)
            print(i, train_acc, test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

    params = {}
    for key in ('W1', 'b1', 'W2', 'b2'):
        params[key] = network.params[key].tolist()

    f = open("./db/param_result/784x50x10-%.5f.json" % test_acc, 'w')
    f.write(json.dumps(params))
    f.close()

    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label='train')
    plt.plot(np.arange(len(test_acc_list)), test_acc_list, label='test')
    plt.legend()
    plt.show()
