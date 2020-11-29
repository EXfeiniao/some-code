import sys
import os
sys.path.append(os.pardir)
from common.functions import *  # functions里所有的函数
from common.gradient import numerical_gradient


# 两层神经网络的类
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重（输出层的神经元数，隐藏层的神经元数，输出层的神经元数）
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)   # 偏置以零初始化
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        '''
        保存神经网络的参数的字典型变量（实例变量）
        params['W1'] 是第一层的权重，params['b1'] 是第一层的偏置
        params['W2'] 是第二层的权重，params['b2'] 是第二层的偏置
        '''

        # 进行识别（推理），参数x是图像数据
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x:输入数据，t：监督数据
    # 计算损失函数的值
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据，t：监督数据
    # 计算权重参数的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        '''
        保存梯度的字典型变量(numerical_gradient()的返回值)
        grads['W1'] 是第一层权重的梯度，grads['b1'] 是第一层偏置的梯度
        grads['W2'] 是第二层权重的梯度，grads['b2'] 是第二层偏置的梯度
        '''

        return grads


if __name__ == '__main__':  # import到其他.py中时，此句子之后的内容不会被执行
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)   # (784, 100)
    print(net.params['b1'].shape)   # (100, )
    print(net.params['W2'].shape)   # (100, 10)
    print(net.params['b2'].shape)   # (10, )
