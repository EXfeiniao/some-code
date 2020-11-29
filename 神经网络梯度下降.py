import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# 神经网络的梯度
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化

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

    # 只有一层
    def predict(self, x):
        return np.dot(x, self.W)  # 返回y = W*x(W是weight)

    # 损失函数（交叉熵）
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = SimpleNet()
print("net.W:")
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print("net.predict(x):")
print(p)
NetMax = np.argmax(p)  # 最大值的索引
print("np.argmax(p):")
print(NetMax)
t = np.array([0, 0, 1])     # 正确解标签
loss = net.loss(x, t)
print(loss)
'''
定义函数时使用了def f(x)
实际上，Python中如果定义的是简单的函数，可以使用lambda表示法：
f = lambda W: net.loss(x,t)
dw = numerical_gradient(f, net.W)
'''
