import numpy as np
# 通过矩阵的乘积进行神经网络的运算

X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])    # weight
print(W)
print(W.shape)
Y = np.dot(X, W)
print("output layer:")
print(Y)
