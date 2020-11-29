import os
import sys
import numpy as np
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist


# 使用load_mnist函数，读入mnist数据集，第一次使用要下载
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)
'''
load_mnist(normalize=False, flatten=True, one_hot_label=False)
第一个参数normalize设置是否将输入图像正规化为0.0-1.0的值
第二个参数flatten设置是否展开输入图像（变成一维数组）
第三个参数one_hot_label设置是否将标签保存为one—hot表示
'''

# 输出各个数据的形状
print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000,10)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(np.random.choice(60000, 10))
