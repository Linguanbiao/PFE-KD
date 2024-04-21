# ！-*- coding:utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from net import CNN_Model, Softmax_Model, CNN_Model_Feature, LFW_CNN_Model, MNIST_CNN_Model
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

# 自己写了一个迭代训练数据的函数
def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]
# 自己写了一个损失函数
'''这个自定义 PyTorch 模块的主要功能是计算带有 L2 正则化的交叉熵损失。
在前向传播中，它首先计算交叉熵损失，然后遍历模型的参数，计算参数的平方和，并将 L2 正则化项添加到损失中。
这样可以在训练过程中对模型的参数进行正则化，以减小过拟合风险。'''
class CrossEntropy_L2(nn.Module):

    def __init__(self, model, m, l2_ratio):

        super(CrossEntropy_L2, self).__init__()
        self.model = model
        self.m = m
        self.w = 0.0
        self.l2_ratio = l2_ratio

    def forward(self, y_pred, y_test):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_test)

        for name in self.model.state_dict():

            if name.find('weight') != -1:
                self.w += torch.sum(torch.square(self.model.state_dict()[name]))

        loss = torch.add(torch.mean(loss), self.l2_ratio * self.w / self.m / 2)

        return loss

def train_model(classifierType, train_x, train_y, test_x, test_y, num_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_in = train_x.shape
    n_out = len(np.unique(train_y))

    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))

    if classifierType == 'cnn':
        n_hidden = 128
        print('Using a multilayer convolution neural network based model...')
        # net = CNN_Model(n_in, n_hidden, n_out)  # cifar10 cifar100  hvfhn GTSRB原始数据用
        # net = CNN_Model_Feature(n_in, n_hidden, n_out)  # cifar10 cifar100  hvfhn GTSRB提取特征后专用
        net = LFW_CNN_Model(n_in, n_hidden, n_out) #lfw 专用
        print(n_in, n_out)  # (1250, 3, 96, 96) 10  (10520, 1, 28, 28) 10
        # net = MNIST_CNN_Model(n_in, n_hidden, n_out)  # Mnist专用
        # net = STL_CNN_Model(n_in, n_hidden, n_out)  # STL 专用
        # net = ResNet18(n_out)  # STL10
        batch_size = 100
        learning_rate = 0.001
        l2_ratio = 1e-07
    else:
        print('Using a single layer softmax based model...')
        net = Softmax_Model(n_in, n_out)
        batch_size = 10
        learning_rate = 0.01
        l2_ratio = 1e-07

    # create loss function
    m = n_in[0]
    criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
    net.to(device)
    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # count loss in an epoch
    temp_loss = 0.0
    # batch count

    print('Training...')
    net.train()

    for epoch in range(num_epoch):
        Itreator = 0  # batch的批次
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # empty parameters in optimizer
            optimizer.zero_grad()

            outputs = net(input_batch)
            # calculate loss value
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
            temp_loss = round(temp_loss, 3)

            # if classifierType == 'cnn' and epoch > (num_epoch * 0.6 ):
            # # print('Epoch {}, train loss {}'.format(epoch, temp_loss))
            #     if(Itreator % 5 > 1):
            #         train_acc , test_acc , _ = test(epoch ,num_epoch, net , train_x, train_y , test_x , test_y , batch_size , device)
            #         if(train_acc > 0.9 or abs(train_acc - test_acc) > 0.4):
            #         # 执行梯度上升
            #             print("执行梯度上升了")
            #             for param in net.parameters():
            #                 if param.grad is not None:
            #                     param.data += 0.1 * param.grad.data
            #         else:
            #             optimizer.step()
            #     else:
            #           optimizer.step()
            # else:
            #     # 前面的一般训练轮数正常训练
            #         optimizer.step()
            # Itreator += 1;

        # train_acc , test_acc = test(epoch , net , train_x, train_y , test_x , test_y , batch_size , device)
        train_acc, test_acc, clf = test(epoch, num_epoch, net, train_x, train_y, test_x, test_y, batch_size, device)
        print('Epoch {}, train loss {}'.format(epoch, temp_loss))
        print('Epoch {}, Train Accuracy {} Test Accuracy {}'.format(epoch, train_acc, test_acc))
        if (epoch == num_epoch - 1):
            print('More detailed results:')
            print(clf)

        temp_loss = 0.0

    return net
def test(epoch, num_epoch, net, train_x, train_y, test_x, test_y, batch_size, device):
    # 获取训练集精度
    net.eval()  # 把网络设定为训练状态
    pred_y = []
    with torch.no_grad():

        for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch)
            input_batch = input_batch.to(device)
            outputs = net(input_batch)
            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
        pred_y = np.concatenate(pred_y)
    train_acc = accuracy_score(train_y, pred_y)

    # 获取测试集精度
    net.eval()
    pred_y = []
    with torch.no_grad():
        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            input_batch = torch.tensor(input_batch)
            input_batch = input_batch.to(device)

            outputs = net(input_batch)

            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

        pred_y = np.concatenate(pred_y)
    test_acc = accuracy_score(test_y, pred_y)
    clf = classification_report(test_y, pred_y)

    # if(epoch == num_epoch -1):
    #     print('More detailed results:')
    #     print(classification_report(test_y, pred_y))

    return train_acc, test_acc, clf


def getTrainModelData(classifierType, targetTrain, targetTrainLabel, targetTest, targetTestLabel, num_epoch):
    # 训练模型
    classifier_net = train_model(classifierType, targetTrain, targetTrainLabel, targetTest, targetTestLabel, num_epoch)
    # 把数据给训练好的模型做查询，得到置信度分数
    attack_x, attack_y = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_net.eval()

    # data used in training, label is 1
    for batch, _ in iterate_minibatches(targetTrain, targetTrainLabel, 100, False):
        batch = torch.tensor(batch)
        batch = batch.to(device)

        output = classifier_net(batch)
        preds_tensor = nn.functional.softmax(output, dim=1)

        attack_x.append(preds_tensor.detach().cpu().numpy())
        attack_y.append(np.ones(len(batch)))

    # data not used in training, label is 0
    for batch, _ in iterate_minibatches(targetTest, targetTestLabel, 100, False):
        batch = torch.tensor(batch)
        batch = batch.to(device)
        output = classifier_net(batch)
        preds_tensor = nn.functional.softmax(output, dim=1)

        attack_x.append(preds_tensor.detach().cpu().numpy())
        attack_y.append(np.zeros(len(batch)))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    return attack_x, attack_y
