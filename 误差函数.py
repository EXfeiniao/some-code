import numpy as np


# 均方误差函数（误差越小越好）
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    '''
    微小值delta
    当出现np.log(0)时，np.log(0)会变成-inf
    这样会导致后续计算无法进行
    '''
    return -np.sum(t * np.log(y + delta))


# mini-batch版交叉熵函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # 当监督数据是标签形式
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
