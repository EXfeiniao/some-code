import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from 显示图像 import img_show, trans

# import sys
# import os
# sys.path.append(os.pardir)


if __name__ == '__main__':
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    '''
    load_mnist(normalize=False, flatten=True, one_hot_label=False)
    第一个参数normalize设置是否将输入图像正规化为0.0-1.0的值
    第二个参数flatten设置是否展开输入图像（变成一维数组）
    第三个参数one_hot_label设置是否将标签保存为one—hot表示
    '''
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    # 28*28 的图像，一层hidden layer

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 梯度
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # loss
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            # 记录学习过程
            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            # print(train_acc, test_acc)
            # str(iter_per_epoch)
            print(' | train acc：{:.4f}, test acc：{:.4f} | loss: ：{:.4f}'.format(train_acc*100, test_acc*100, loss))


    # 显示图像
    img = x_train[i]*255
    label = np.argmax(t_train[i])
    # label = t_train[i][7]
    print(' | label: {:} '.format(label))

    # print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 将图像的形状变成原来的尺寸
    # print(img.shape)  # (28, 28)

    img_show(img)

