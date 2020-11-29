import numpy as np
# softmax函数的实现


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a   # 广播

    return y


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
