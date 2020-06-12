#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/01/01

import numpy as np
import matplotlib.pyplot as plt

from mnist import load_mnist
from multi_layer_net import MultiLayerNet
from trainer import Trainer

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    # 减少学习数据, 造成过拟合现象
    x_train = x_train[:300]
    t_train = t_train[:300]

    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            dropout_ration=0.2)
    trainer = Trainer(network,
                      x_train,
                      t_train,
                      x_test,
                      t_test,
                      epochs=301,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': 0.01},
                      verbose=True)

    for flag in [False, True]:
        network.set_dropout(flag)
        trainer.set_network(network)
        trainer.train()
        train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

        # 绘制图形==========
        plt.subplot(1, 2, flag + 1)
        plt.title("use_dropout = " + str(flag))
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_acc_list))
        plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
        plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
    plt.show()
