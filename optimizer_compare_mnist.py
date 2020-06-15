#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/06/05

import numpy as np
import matplotlib.pyplot as plt

from mnist import load_mnist
from optimizer import SGD, Momentum, AdaGrad, Adam
from multi_layer_net import MultiLayerNet
from utils import smooth_curve


if __name__ == "__main__":
    #  读取 mnist 数据
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=True)
    train_size = train_img.shape[0]
    batch_size = 128
    iters_times = 2000

    # 初始化
    optimizers = {}
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10)
        train_loss[key] = []

    # 训练网络
    for i in range(iters_times):
        batch_mask = np.random.choice(train_size, batch_size)
        train_img_batch = train_img[batch_mask]
        train_label_batch = train_label[batch_mask]

        for key in optimizers.keys():
            # 计算梯度 更新参数
            grads = networks[key].gradient(train_img_batch, train_label_batch)
            optimizers[key].update(networks[key].params, grads)

            # 计算 loss
            loss = networks[key].loss(train_img_batch, train_label_batch)
            train_loss[key].append(loss)

        # 每 100 次打印 1 次 loss
        if i % 100 == 0:
            print("=========== " + "iteration: " + str(i) + " ===========")
            for key in optimizers.keys():
                print(key + ": " + str(train_loss[key][-1]))

    # 绘制曲线
    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
    x = np.arange(iters_times)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
