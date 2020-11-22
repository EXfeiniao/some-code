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

        # 对每层的定义
        '''
        Linear函数：线性模块。
        Linear(1,2) 第一个参数是input的个数或者上一层的神经元的个数
        第二个参数是本层神经元的个数或者output的个数
        Linear函数源代码：
        class Linear(Module)：
            __constants__ = ['bias']
        def __init__(self, in_features, out_features, bias=True):
            super(Linear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            # Parameter函数：将一个不可训练的类型Tensor转换成可以训练的类型parameter
            并将这个parameter绑定到这个module里面
            if bias:
                self.bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
        需要实现的内容：y = xA^T + b
        通过Linear函数改变张量大小？
        batch_size
        '''
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
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        y = F.softmax(self.fc5(x), dim=1)
        '''
        softmax类似给出每个值在一组值中的比例
        si = ei/sum（ej）（其中，e为自然对数，i为输入值，且为e的指数）
        dim 这个参数是维度
        '''
        return y
