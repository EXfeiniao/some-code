import torch
import torch.nn as nn  # 导入torch库的nn，且此后调用其中函数直接用nn
import torch.nn.functional as F  # 导入torch库的functional，且此后调用其中函数直接用F

import os  # 导入操作系统接口模块


class DNN(nn.Module):  # Python类
    # 在__init__构造函数中声明各个层的定义
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()  # 调用的顺序：DNN -> self

        # 实例变量 self.**
        self.input_size = input_size
        self.output_size = output_size

        # 对每层的定义(矩阵内积)
        self.fc1 = nn.Linear(input_size, 5120)
        self.fc2 = nn.Linear(5120, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(512, output_size)

    # 在forward中实现层之间的连接关系，实际就是前向传播的过程
    def forward(self, x):
        x = x.view(-1, self.input_size)
        # view函数可以改变tensor的形状
        x = F.leaky_relu(self.fc1(x))
        '''
        leaky_relu函数是一个激活函数，
        sigmoid和tanh是“饱和激活函数”，而ReLU及其变体则是“非饱和激活函数”。
        使用“非饱和激活函数”的优势在于两点：
        1.首先，“非饱和激活函数”能解决所谓的“梯度消失”问题。
                                    哦哦，原来如此！
        2.其次，它能加快收敛速度。
        Sigmoid函数需要一个实值输入压缩至[0,1]的范围
        σ(x) = 1 / (1 + exp(−x))
        tanh(x) = 2σ(2x) − 1
        Leaky ReLUs
        ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。
        Leaky ReLU激活函数是在声学模型（2013）中首次提出的。以数学的方式我们可以
        表示为：
        yi = { xi if xi >= 0 & xi/ai if xi < 0 }
        '''
        
        # 循环计算x，实现forward
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        y = F.softmax(self.fc5(x), dim=1)
        return y
