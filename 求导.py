import numpy as np
import matplotlib.pylab as plt


# 数值微分
def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h)) - f(x-h) / (2*h)


# 绘制函数
def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1)   # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# 求x = 5，x = 10 处的函数（有误）
# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))
