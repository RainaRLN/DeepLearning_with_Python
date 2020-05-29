#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/26

import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t, onehot=True):
    if y.ndim == 1:  # 若维度为1，转换为2维， 兼容批量数据处理
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    delta = 1e-7  # 加上一个很小值, 避免出现 log(0)
    if onehot:
        return -np.sum(t * np.log(y + delta)) / batch_size
    else:
        return -np.sum(np.log(y[np.arrange(batch_size), t] + delta)) / batch_size


if __name__ == "__main__":
    pass

