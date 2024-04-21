import warnings
import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
# from featureExtractorNet.cnn import cnnNet
# from featureExtractorNet.leakcnn import LeaksCNN
warnings.filterwarnings("ignore", category=UserWarning)

# 设置图片的预处理操作

# device 设置为 GPU, 加载预训练模型 , VGG16 分为两部分，一部分是特征提取器，一部分是分类器


def loadData():
    # 图像数据的处理方式  50000+10000 32 32 3
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize([224, 224]),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # # 加载CIFAR10数据集 ， 返回的是一个 tuple 数据类型 （image , label)
    # ds_train = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True, download=True, transform=transform)
    # ds_test = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=False, download=True, transform=transform)

    # CIFAR100 图像数据的处理方式
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel image
    #     transforms.Resize((224, 224)),  # Resize images to the size expected by ResNet
    #     transforms.ToTensor(),
    # ])
    # # 加载数据集 ， 返回的是一个 tuple 数据类型 （image , label)
    # ds_train = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=True, download=True, transform=transform)
    # ds_test = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=False, download=True, transform=transform)

    # 加载MNIST数据集 60000+10000 28 28 1
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel image
    #     transforms.Resize((224, 224)),  # Resize images to the size expected by ResNet
    #     transforms.ToTensor(),
    # ])
    #
    # ds_train= torchvision.datasets.FashionMNIST(root='./dataset', train=True, transform=transform, download=True)
    # ds_test  = torchvision.datasets.FashionMNIST(root='./dataset', train=False, transform=transform, download=True)


    # SVHN 图像数据的处理方式  73,257+26,032 32 32 3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载SVHN数据集 ， 返回的是一个 tuple 数据类型 （image , label)
    ds_train =torchvision.datasets.SVHN(root='./dataset/SVHN', split='train', transform=transform, download=True)
    
    ds_test = torchvision.datasets.SVHN(root='./dataset/SVHN', split='test', transform=transform, download=True)

    # GTSRB 图像数据的处理方式
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel image
    #     transforms.Resize((224, 224)),  # Resize images to the size expected by ResNet
    #     transforms.ToTensor(),
    # ])
    # # 加载数据集 ， 返回的是一个 tuple 数据类型 （image , label)
    # ds_train = torchvision.datasets.GTSRB('./dataset', split='train', transform=transform, download=True)
    # ds_test = torchvision.datasets.GTSRB('./dataset', split='test', transform=transform, download=True)

    # cluster = 10000
    batch_size = 128
    # target_train = Subset(ds_train, range(cluster))

    targetTrainDataLoader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    targetTestDataLoader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    return targetTrainDataLoader, targetTestDataLoader


# 定义模型，特征提取器, 是指 VGG16 网络最后一个 conv 层进行特征提取
def makePreTrainedModel():
    # VGG16 Cifar10 SVHW提取到的图片特征尺寸大小为：(50000, 16, 7, 7) , 标签的尺寸大小是：(50000,)
    # model = models.vgg16(pretrained=True).features[:]  # 其实就是定位到第 31 层,最后一层卷积层
    # model[28] = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # VGG19  Cifar10 SVHW提取到的图片特征尺寸大小为：(50000, 16, 7, 7) , 标签的尺寸大小是：(50000,)(73257, 16, 7, 7)
    # model = models.vgg19(pretrained=True).features[:]
    # model[34] = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # VGG19  Cifar10 SVHW 提取到的图片特征尺寸大小为：(50000, 32, 7, 7) , 标签的尺寸大小是：(50000,) ,
    model = models.vgg19(pretrained=True).features[:]
    model[34] = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # ResNet18  Cifar10 SVHW提取到的图片特征尺寸大小为：(50000, 32, 7, 7) , 标签的尺寸大小是：(50000,)
    # model = models.resnet18(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    # # ResNet18 SVHW提取到的图片特征尺寸大小为：(32,8,8) (32,9,9)
    # model = models.resnet18(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2), bias=False)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    # # ResNet18 SVHW提取到的图片特征尺寸大小为：(16,13,13)
    # model = models.resnet18(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # # 修改提取到的特征尺寸
    # model[6][0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[6][0].bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[6][0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
    # model[7][0].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # custom_conv = nn.Conv2d(32, 16, kernel_size=3, stride=1 , padding=1)
    # # 构建新的模型
    # model = nn.Sequential(
    #     nn.Sequential(*model),
    #     custom_conv
    # )


    # ResNet34  Cifar10 SVHW 提取到的图片特征尺寸大小为：(50000, 32, 7, 7) , 标签的尺寸大小是：(50000,)
    # model = models.resnet34(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][2].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    # ResNet50 Cifar10 SVHW 提取到的图片特征尺寸大小为：(50000, 32, 7, 7) , 标签的尺寸大小是：(50000,)
    # model = models.resnet50(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(1024, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][2].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    # VGG19 MNIST  提取到的图片特征尺寸大小为：(60000, 1, 14, 14)
    # 加载预训练的VGG19模型
    # model = models.vgg19(pretrained=True)
    # # 去掉VGG19的全连接层 去掉最后的池化层，以获得更大的特征图
    # features = nn.Sequential(*list(model.features)[:-1])
    # # 添加一个自定义的卷积层来获得1个通道的特征图
    # custom_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)
    # # 构建新的模型
    # model = nn.Sequential(
    #     features,
    #     custom_conv
    # )

    # VGG16 MNIST  提取到的图片特征尺寸大小为：(60000, 1, 14, 14)
    # 去掉VGG16的全连接层和最后的池化层
    # features = list(models.vgg16(pretrained=True).features)[:-1]
    # # 添加一个自定义的卷积层来获得1个通道的特征图
    # custom_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)
    # # 构建新的模型
    # model = nn.Sequential(
    #     nn.Sequential(*features),
    #     custom_conv
    # )

    # ResNet18  MNIST  提取到的图片特征尺寸大小为：(60000, 1, 14, 14)
    # model = models.resnet18(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # custom_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    # # 构建新的模型
    # model = nn.Sequential(
    #     nn.Sequential(*model),
    #     custom_conv
    # )

    # ResNet34  MNIST  提取到的图片特征尺寸大小为：(60000, 1, 14, 14)
    # model = models.resnet34(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][2].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].conv2 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # custom_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    # # 构建新的模型
    # model = nn.Sequential(
    #     nn.Sequential(*model),
    #     custom_conv
    # )

    # # ResNet50 MNIST (60000, 1, 14, 14)
    # model = models.resnet50(pretrained=True)
    # model = nn.Sequential(*list(model.children())[:-2])
    # # 修改提取到的特征尺寸
    # model[7][0].downsample[0] = nn.Conv2d(1024, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].downsample[1] = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][0].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][0].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][1].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][1].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # model[7][2].conv1 = nn.Conv2d(32, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].conv3 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model[7][2].bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # custom_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    # # 构建新的模型
    # model = nn.Sequential(
    #     nn.Sequential(*model),
    #     custom_conv
    # )

    # # VGG16 MNIST  提取到的图片特征尺寸大小为：(60000, 1, 14, 14)
    # # 去掉VGG16的全连接层和最后的池化层
    # model = models.vgg16(pretrained=True).features[:]  # 其实就是定位到第 31 层,最后一层卷积层
    # model[23] = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)
    # model[28] = nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)
    #
    # 输入随机数据以查看输出形状
    x = torch.randn(1, 3, 224, 224)  # 输入数据的形状通常是 3xHxW
    output = model(x)
    print(output.shape)  #（1，32，7，7
    print(model)
    model = model.eval()    # 调整模型为测试状态
    return model

# 定义特征提取的方法


def featuresExtract(model, dataLoader, device, dataType):
    model.eval()  # 将网络调到测试状态aLoader = dataLoader
    # print("特征提取模型结构为: ", model)
    with torch.no_grad():
        for step, (features, labels) in enumerate(dataLoader):
            features = features.to(device)
            labels = labels.to(device)
            # 特征提取
            features_result = model(Variable(features))
            trainX = features_result.cpu().detach().numpy()
            trainY = labels.cpu().detach().numpy()
            if(step == 0):
                trainDataFeatures = trainX
                trainDataLabels = trainY
                continue
            trainDataFeatures = np.concatenate((trainDataFeatures, trainX), axis=0)
            trainDataLabels = np.concatenate((trainDataLabels, trainY), axis=0)
            print("特征提取中: 请稍等")
        print("提取到的图片特征尺寸大小为：{} , 标签的尺寸大小是：{} , 数据的类型是：{}".format(
            trainDataFeatures.shape, trainDataLabels.shape, type(trainDataFeatures)))
    if(dataType == 0):
        # type = 0  表示是训练数据做特征提取， type = 1 表示是测试数据做特征提取
        np.savez('./data/featureExtracted/targetTrainData.npz', features = trainDataFeatures, labels = trainDataLabels)
    else:
        np.savez('./data/featureExtracted/targetTestData.npz', features = trainDataFeatures, labels =  trainDataLabels)


def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_pred_cls = y_pred_cls.cpu().detach().numpy()   # 将 tensor 转换成 numpy 形式
    y_true = y_true.cpu().detach().numpy()
    return accuracy_score(y_pred_cls, y_true)


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
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(
            start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def load_data(data_name):
    with np.load(data_name) as data:
        features = data['features']  # 假设特征数组的键为 'features'
        labels = data['labels']  # 假设标签数组的键为 'labels'
    return features, labels


# def train(device):
    data_path = './data/featureExtracted'
    trainX, trainY = load_data(data_path + '/targetTainData.npz')
    testX, testY = load_data(data_path + '/targetTestData.npz')
    # print("查看标签是什么----", testY[:100])
    print("训练数据的尺寸是:{} , 标签的尺寸是:{} , 数据类型是:{}".format(trainX.shape, trainY.shape, type(trainX)))
    print("测试数据的尺寸是:{} , 标签的尺寸是:{} , 数据类型是:{}".format(testX.shape, testY.shape, type(testX)))
    net = cnnNet()
    # net = LeaksCNN()
    net.to(device)
    cross_loss = nn.CrossEntropyLoss()
    optimezer = optim.Adam(params=net.parameters(), lr=0.0003)
    # optimezer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.5)
    metric_fuc = accuracy
    epochs = 50
    batch_size = 256
    print("Train...")
    net.train()
    for epoch in range(epochs):
        # 训练循环
        steps = 1
        loss_sum = 0.0
        metric_sum = 0.0
        for features_batch, labels_batch in iterate_minibatches(trainX, trainY, batch_size):
            features_batch, labels_batch = torch.tensor(features_batch), torch.tensor(labels_batch).type(torch.long)
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            # 梯度置 0
            optimezer.zero_grad()
            # 前向传播
            prediction_y = net(features_batch)
            loss = cross_loss(prediction_y, labels_batch)
            # 反向传播
            loss.backward()
            optimezer.step()
            acc = metric_fuc(prediction_y, labels_batch)
            loss_sum += loss.item()
            metric_sum += acc.item()

            if(steps % 10 == 0):
                print("Epoch {} / iteration {} , train loss {} , train accuracy{} ".format(epoch,
                                                                                           steps, loss_sum / steps, metric_sum / steps))
            steps += 1
        # 测试循环
        val_steps = 1
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        net.eval()  # 把模型调成测试状态
        for features_batch, labels_batch in iterate_minibatches(testX, testY, batch_size):
            features_batch, labels_batch = torch.tensor(features_batch), torch.tensor(labels_batch).type(torch.long)
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            with torch.no_grad():
                prediction_y = net(features_batch)
                loss = cross_loss(prediction_y, labels_batch)
                test_acc = metric_fuc(prediction_y, labels_batch)
                val_loss_sum += loss.item()
                val_metric_sum += test_acc.item()
                val_steps += 1
        print("[Epoch {} / Epochs{}] -------- Test loss {} -------- Test accuracy {} ".format(epoch,
              epochs, val_loss_sum / val_steps, val_metric_sum / val_steps))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    isFirstFeaturesExtract = True
    model = makePreTrainedModel()
    model = model.to(device)
    if(isFirstFeaturesExtract):
        targetTrainDataLoader,  targetTestDataLoader = loadData()
        featuresExtract(model, targetTrainDataLoader, device, 0)   # 训练数据的特征提取
        # featuresExtract(model, targetTestDataLoader, device, 1)    # 测试数据的特征提取
    # train(device)
