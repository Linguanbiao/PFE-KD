
from torch import nn
from torchvision import models

class CNN_Model(nn.Module): # (3, 32, 32)
    def __init__(self, n_in, n_hidden, n_out):

        super(CNN_Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in[1],
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=2,
                      stride=1),
            nn.ReLU()
        )

        self.maxp1 = nn.MaxPool2d(
            kernel_size=(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=0,
                      stride=1),
            nn.ReLU()
        )

        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Sequential(
            # nn.Linear(in_features=32 * 29 * 21 , out_features=n_hidden),        # Face
            # nn.Linear(in_features=32 * 5 * 5, out_features=n_hidden),  # Mnist
            nn.Linear(in_features=32 * 6 * 6, out_features=n_hidden),  # cifar 10 or cifar 100
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_out)
        )

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class CNN_Model_Feature(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(CNN_Model_Feature, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in[1],
                      out_channels=32,
                      kernel_size=(3, 3),
                      padding=1,
                      stride=1),
            nn.ReLU()
        )

        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3, 3),
                      padding=1,
                      stride=1),
            nn.ReLU()
        )

        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32*1*1, out_features=n_hidden),
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_out)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class NN_Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(NN_Model, self).__init__()

        # 计算输入的展平维度
        input_size = n_in[1] * n_in[2] * n_in[3]

        # 添加一个全连接层来处理展平后的输入
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=n_hidden),
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_out)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平操作，将输入变换为(batch_size, input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class LFW_CNN_Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):

        super(LFW_CNN_Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in[1],
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=2,
                      stride=1),
            nn.ReLU()
        )

        self.maxp1 = nn.MaxPool2d(
                       kernel_size=(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=0,
                      stride=1),
            nn.ReLU()
        )

        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32 * 29 * 21 , out_features=n_hidden),          # LFW
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_out)
        )

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class MNIST_CNN_Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):

        super(MNIST_CNN_Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in[1],
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=2,
                      stride=1),
            nn.ReLU()
        )

        self.maxp1 = nn.MaxPool2d(
                       kernel_size=(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=0,
                      stride=1),
            nn.ReLU()
        )

        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32 * 1 * 1, out_features=n_hidden),          # MNIST 原始32 * 5 * 5 抽取过32 * 1* 1
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_out)
        )

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Softmax_Model(nn.Module):

    def __init__(self, n_in, n_out):

        super(Softmax_Model, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=n_in[1],
                      out_features=n_out)
        )

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        x = self.fc1(x)

        return x