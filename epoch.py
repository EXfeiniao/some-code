import numpy as np
import os
import sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
# 从dataset.mnist 调用加载mnist数据集的方法
from two_layers_net import TwoLayerNet
# 从two_layers_net调用类TwoLayerNet


# 使用load_mnist函数，读入mnist数据集，第一次使用要下载
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
'''
load_mnist(normalize=False, flatten=True, one_hot_label=False)
第一个参数normalize设置是否将输入图像正规化为0.0-1.0的值
第二个参数flatten设置是否展开输入图像（变成一维数组）
第三个参数one_hot_label设置是否将标签保存为one—hot表示
'''

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
