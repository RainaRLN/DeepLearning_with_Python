#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/06/11

import numpy as np
import matplotlib.pyplot as plt

from mnist import load_mnist
from multi_layer_net import MultiLayerNet
from optimizer import SGD


def train(weight_init_std, x_train, t_train, max_epochs):
    batch_norm_network = MultiLayerNet(input_size=784,
                                       hidden_size_list=[100, 100, 100, 100, 100],
                                       output_size=10,
                                       weight_init_std=weight_init_std,
                                       use_batchnorm=True)
    no_batch_norm_network = MultiLayerNet(input_size=784,
                                          hidden_size_list=[100, 100, 100, 100, 100],
                                          output_size=10,
                                          weight_init_std=weight_init_std)
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01
    max_iters_times = 1000000000
    epoch = max(int(train_size / batch_size), 1)

    optimizer = SGD(lr=learning_rate)
    bn_train_acc_list = []
    no_bn_train_acc_list = []

    epoch_cnt = 0
    for i in range(max_iters_times):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for network in (batch_norm_network, no_batch_norm_network):
            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

        if i % epoch == 0:
            bn_train_acc = batch_norm_network.accuracy(x_train, t_train)
            no_bn_train_acc = no_batch_norm_network.accuracy(x_train, t_train)
            bn_train_acc_list.append(bn_train_acc)
            no_bn_train_acc_list.append(no_bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(no_bn_train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return no_bn_train_acc_list, bn_train_acc_list


if __name__ == "__main__":
    # FIXME: Runtimewarnings
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    print(x_train.shape)
    # 减少学习数据
    x_train = x_train[:1000]
    t_train = t_train[:1000]

    weight_scale_list = np.logspace(0, -4, num=16)  # 生成不同weight
    max_epochs = 20
    x = np.arange(max_epochs)  # x轴

    for i, w in enumerate(weight_scale_list):
        print("============== " + str(i + 1) + " / 16" + " ==============")
        no_bn_train_acc_list, bn_train_acc_list = train(w, x_train, t_train, max_epochs)

        plt.subplot(4, 4, i + 1)
        plt.title("W: " + str(w))
        if i == 15:
            plt.plot(x, bn_train_acc_list, label="Batch Normalization", markevery=2)
            plt.plot(x, no_bn_train_acc_list, linestyle='--', label="Without Batch Normalization", markevery=2)
        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, no_bn_train_acc_list, linestyle='--', markevery=2)

        plt.ylim(0, 1.0)

        if i % 4:  # 除最左列, 其他取消显示y轴刻度
            plt.yticks([])
        else:  # 最左列添加ylabel
            plt.ylabel("accuracy")

        if i < 12:
            plt.xticks([])
        else:
            plt.xlabel("epoch")

        plt.legend(loc="lower right")

    plt.show()
