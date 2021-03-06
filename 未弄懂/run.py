import torch    # 导入torch库
import torch.nn as nn   # 导入torch库的nn，且此后调用其中函数直接用nn
import torch.nn.functional as F
# 导入torch库的functional，且此后调用其中函数直接用F
import numpy as np  # 导入numpy库，且此后调用其中的函数直接用np

from options import get_args
from data import get_data
from util import get_grid
import os   # 导入操作系统接口模块

from model import DNN   # 从model.py导入DNN类

if __name__ == '__main__':   # 使得run.py import到其他.py中时，此句子之后的内容不会被执行
    args = get_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # 异常处理
    try:
        os.makedirs(args.output)
    except OSError:
        pass
    try:
        os.makedirs(args.log)
    except OSError:
        pass
    
    # 加载训练数据和测试数据
    train_loader, test_loader = get_data(args)
    
    # 如果有cuda就用cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 28*28
    output_size = args.num_classes
    model = DNN(input_size=input_size, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 每训练完一个mini-batch就计算一次loss，acc
    model.train()
    for epoch in range(args.epochs):
        correct = 0
        total = 0
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, y_pred_t = torch.max(y_pred.data, 1)
            total += y.size(0)
            # print(y_pred_t, y.data)
            # print((y_pred_t == y).sum().item())
            correct += (y_pred_t == y).sum().item()
            # print(correct, total)

            loss = criterion(y_pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx + 1) % 100 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                      .format(epoch + 1, args.epochs, idx + 1, len(train_loader), loss.item(), 100 * correct / total))
    
    # 保存模型参数
    torch.save(model.state_dict(), os.path.join('./log', '{}_{}_{}.ckpt'.format(args.model, args.dataset, args.epochs)))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, y_pred = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (y_pred == y).sum().item()
            # print(result)
            if idx % 100 == 0:
                get_grid(x.cpu().numpy(), args, args.epochs, idx)
                print(y_pred.data.cpu().numpy(), y.data.cpu().numpy)
        print('Test Acc: {:.4f}%, Model: {}, Epochs: {}'.format(correct/total*100, args.model, args.epochs))




