#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/06/08

import numpy as np
from collections import OrderedDict

from layers import Sigmoid, Affine, Relu, SoftmaxWithLoss


class MultiLayerNet:
    """
    多层神经网络
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        """
        :param input_size: 输入层大小
        :param hidden_size_list: 隐藏层神经元数量的列表 (e.g. [50, 50, 50])
        :param output_size: 输出层大小
        :param activation: 激活函数 ('relu' or 'sigmod')
        :param weight_init_std: 指定权重标准差 (e.g. 0.01)
                指定 'relu' 或 'he': 使用 "He 初始值"
                指定 'sigmoid' 或 'xavier: 使用 "Xavier 初始值"
        :param weight_decay_lambda: Weight Decay (L2 范数) 的强度
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 初始化层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],
                                                    self.params['b'+str(idx)])
            self.layers['activation_function'+str(idx)] = activation_layer[activation]()
        
        idx += 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],
                                                self.params['b'+str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])

            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layers in self.layers.values():
            x = layers.forward(x)
        
        return x

    def loss(self, x, t):
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc
    
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
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = self.layers['Affine'+str(idx)].dW + self.weight_decay_lambda * self.layers['Affine'+str(idx)].W
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db

        return grads


if __name__ == "__main__":
    pass
