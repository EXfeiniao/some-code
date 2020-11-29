import numpy as np


# 数值微分
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h)) - f(x - h) / (2 * h)


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_diff(f, x)
        x -= lr * grad

    return x


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
array = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(array)
