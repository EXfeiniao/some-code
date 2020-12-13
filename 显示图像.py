# 显示图像
import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def trans(x):
    for i in range(10):
        if x[i] == 1:
            # print(x[i])
            return i


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img), mode='L')
    pil_img.show()


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]
    label = t_train[0]
    print(label)    # 5

    print(img.shape)            # (784,)
    img = img.reshape(28, 28)   # 将图像的形状变成原来的尺寸
    print(img.shape)            # (28, 28)

    img_show(img)
