import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist


# 使用load_mnist函数，读入mnist数据集，第一次使用要下载
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
'''
load_mnist(normalize=False, flatten=True, one_hot_label=False)
第一个参数normalize设置是否将输入图像正规化为0.0-1.0的值
第二个参数flatten设置是否展开输入图像（变成一维数组）
第三个参数one_hot_label设置是否将标签保存为one—hot表示
'''

# 输出各个数据的形状
print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000,)
print(x_test.shape)    # (10000, 784)
print(t_test.shape)    # (10000,)
