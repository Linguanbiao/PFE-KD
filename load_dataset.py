# ！-*- coding:utf-8 -*-

import pickle
import numpy as np
from sklearn import model_selection,datasets
import sklearn
import torchvision
from torch.utils.data import Subset
from torchvision import datasets, transforms

def readCIFAR10():
    data_path ='./dataset/CIFAR10/cifar-10-batches-py'
    for i in range(5):
        f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
        train_data_dict = pickle.load(f, encoding='iso-8859-1')
        f.close()
        if i == 0:
            X = train_data_dict["data"]
            y = train_data_dict["labels"]
            continue
        X = np.concatenate((X, train_data_dict["data"]), axis=0)
        y = np.concatenate((y, train_data_dict["labels"]), axis=0)
    f = open(data_path + '/test_batch', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()
    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["labels"])
    print(X.shape)
    return X, y, XTest, yTest
def readMINST():
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST('./dataset/MNIST', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST('./dataset/MNIST', train=False, transform=transform, download=True)

    # 获取数据和标签
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()

    return train_data , train_labels ,test_data ,test_labels

def readFASHIONMINST():
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载MNIST数据集
    train_dataset = datasets.FashionMNIST('./dataset', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST('./dataset', train=False, transform=transform, download=True)

    # 获取数据和标签
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()
    print('train_data====type', type(train_data))
    return train_data , train_labels ,test_data ,test_labels

def readGTSRB():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Resize((32, 32)),  # 调整图像大小为32x32
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化操作
    ])
    # 加载flowers数据集
    train_dataset = datasets.GTSRB('./dataset', split='train', transform=transform, download=True)
    test_dataset = datasets.GTSRB('./dataset', split='test', transform=transform, download=True)

    # 将数据集转换为NumPy数组的函数
    def dataset_to_numpy(dataset):
        num_samples = len(dataset)
        all_data = []
        all_labels = []
        for i in range(num_samples):
            data, label = dataset[i]
            # 检查数据的形状是否正确
            if data.shape == (3, 32, 32):
                all_data.append(data.numpy())
                all_labels.append(label)
        return np.array(all_data), np.array(all_labels)
    # 调用函数转换训练集和测试集
    train_data, train_labels = dataset_to_numpy(train_dataset)
    test_data, test_labels = dataset_to_numpy(test_dataset)
    print('train_data.shape===', train_data.shape)
    return train_data, train_labels, test_data, test_labels

def readFood101():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Resize((32, 32)),  # 调整图像大小为32x32
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化操作
    ])
    # 加载flowers数据集
    train_dataset = datasets.Food101('./dataset', split='train', transform=transform, download=True)
    test_dataset = datasets.Food101('./dataset', split='test', transform=transform, download=True)

    # 将数据集转换为NumPy数组的函数
    def dataset_to_numpy(dataset):
        num_samples = len(dataset)
        all_data = []
        all_labels = []
        for i in range(num_samples):
            data, label = dataset[i]
            # 检查数据的形状是否正确
            if data.shape == (3, 32, 32):
                all_data.append(data.numpy())
                all_labels.append(label)
        return np.array(all_data), np.array(all_labels)
    # 调用函数转换训练集和测试集
    train_data, train_labels = dataset_to_numpy(train_dataset)
    test_data, test_labels = dataset_to_numpy(test_dataset)
    print('train_data.shape===', train_data.shape)
    return train_data, train_labels, test_data, test_labels
def readCIFAR100():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Resize((32, 32)),  # 调整图像大小为32x32
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化操作
    ])

    # 加载CIFAR-100数据集
    train_dataset = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=True, transform=transform,
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=False, transform=transform,
                                                 download=True)
    # 将数据集转换为NumPy数组
    def dataset_to_numpy(dataset):
        num_samples = len(dataset)
        all_data = np.empty((num_samples, 3, 32, 32), dtype=np.float32)
        all_labels = np.empty((num_samples,), dtype=np.int64)

        for i in range(num_samples):
            data, label = dataset[i]
            all_data[i] = data.numpy()
            all_labels[i] = label

        return all_data, all_labels

    train_data, train_labels = dataset_to_numpy(train_dataset)
    test_data, test_labels = dataset_to_numpy(test_dataset)

    print('train_data.shape===',train_data.shape)

    return train_data, train_labels, test_data, test_labels

def readSVHN():  # (73257, 3, 32, 32)
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Resize((32, 32)),  # 调整图像大小为32x32
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化操作
    ])

    # 下载并加载SVHN训练集
    train_dataset = torchvision.datasets.SVHN(root='./dataset/SVHN', split='train', transform=transform, download=True)

    # 下载并加载SVHN测试集
    test_dataset = torchvision.datasets.SVHN(root='./dataset/SVHN', split='test', transform=transform, download=True)

    # 将数据集转换为NumPy数组的函数
    def dataset_to_numpy(dataset):
        num_samples = len(dataset)
        all_data = []
        all_labels = []

        for i in range(num_samples):
            data, label = dataset[i]
            # 检查数据的形状是否正确
            if data.shape == (3, 32, 32):
                all_data.append(data.numpy())
                all_labels.append(label)
        return np.array(all_data), np.array(all_labels)

    # 调用函数转换训练集和测试集
    train_data, train_labels = dataset_to_numpy(train_dataset)
    test_data, test_labels = dataset_to_numpy(test_dataset)

    print('train_data.shape===', train_data.shape)
    return train_data, train_labels, test_data, test_labels
def readSTL():
    # 定义数据预处理的转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像数据转换为PyTorch张量
    ])
    # 加载STL-10数据集，如果不存在将自动下载
    train_dataset = torchvision.datasets.STL10(root='./dataset/STL10', split='train', transform=transform,
                                                  download=True)
    test_dataset = torchvision.datasets.STL10(root='./dataset/STL10',split='test', transform=transform,
                                                 download=True)
    # 获取数据和标签
    train_data = train_dataset.data  # 训练集数据
    train_labels = np.array(train_dataset.labels)  # 训练集标签
    test_data = test_dataset.data  # 测试集数据
    test_labels = np.array(test_dataset.labels)  # 测试集标签

    return train_data , train_labels ,test_data ,test_labels
def readLFW():
    data_path = './dataset/LFW/lfw_funneled'
    lfw_people = sklearn.datasets.fetch_lfw_people(data_home=data_path, min_faces_per_person=40, resize=1)
    n_samples, h, w = lfw_people.images.shape  # resize=0.4 (1867,50,37)  # resize=1.0 (1867, 125, 94)
    print("n_samples: %d" % n_samples)
    x = lfw_people.images.reshape(n_samples, 1, h, w) / 255.0
    y = lfw_people.target
    trainX, testX, trainY, testY = model_selection.train_test_split(x, y, test_size=.1, random_state=42)
    print("trainX------", trainX.shape)
    print("testX------", testX.shape)
    return trainX, trainY, testX, testY  # trainX(1680,1,50,37),testX(187,1,50,37)

def readFeatureExtractedCIFAR10():

    """
    读取一个 npz 文件，并获取其中的特征和标签数组

    参数:
    filepath (str): npz 文件的路径

    返回:
    np.ndarray: 特征数组
    np.ndarray: 标签数组
    """
    filepath = 'data/featureExtracted/targetTrainData.npz'
    with np.load(filepath) as data:
        features = data['features']  # 假设特征数组的键为 'features'
        labels = data['labels']      # 假设标签数组的键为 'labels'
        
    return features, labels

