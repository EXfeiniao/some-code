import os
import sys
import pickle
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


# sigmoid函数的实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax函数的实现
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # 溢出对策
    '''
    计算机运行的缺陷：如果是exp1000这样特别大数字，计算机会返回一个表示
    无穷大的inf，如果在超大值之间进行除法运算，结果就会出现不确定的情况。
    p66：softmax函数在运算时加上和减去一个常数并不会影响结果，所以我们
    减去信号中的最大值来防止溢出
    '''
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a   # 广播

    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return a3


x, t = get_data()
network = init_network()


accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)    # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("accuracy::" + str(float(accuracy_cnt) / len(x)))
