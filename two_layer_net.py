import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 用高斯分布进行初始化
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        '''
        高斯分布一般指正态分布。
        之所以使用正态分布进行初始化：
        正态分布有个3σ原则，意思就是99.7%的点都在距离中心3个标准差之内。
        换句话说，随机初始化的权值依然是有可能是离中心点比较远的。
        假设我们用了sigmoid作为激活函数，一旦初始值过小，或者过大，就可能会导致
        输入到隐藏层的数据太小或者太大，从而进入饱和区。一旦进入饱和区，这些对应
        的neuron就死掉了，再也不会更新了。
        所以为了防止这个dying neuron现象，我们就用了截断正态，保证初始的权值
        不太大也不太小。
        ps:
        如果是sigmoid激活函数，截断正态分布可以防止梯度消失
        如果是relu激活函数，截断正态分布可以防止一上来很多神经元就死掉
        '''

        # 生成层
        self.layers = OrderedDict()
        '''
        OrderedDict: 有序字典
        “有序”是指他可以记住向字典里添加元素的顺序。
        因此，神经网络的正向传播只需按照添加元素的顺序调用各层即可。
        '''
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
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
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
