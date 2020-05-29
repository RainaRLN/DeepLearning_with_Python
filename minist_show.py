#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Raina
# Date: 2020/05/26

from PIL import Image
from minist import load_minist
import numpy as np


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == "__main__":
    (train_img, train_label), (test_img, test_label) = load_minist(normalize=False, flatten=True, one_hot_label=False)
    img = train_img[3].reshape(28, 28)
    label = train_label[3]
    print(label)

    img_show(img)