
import numpy as np
from load_dataset import readCIFAR10, readMINST, readFeatureExtractedCIFAR10, readCIFAR100, readLFW, readSTL, readSVHN, \
    readFASHIONMINST, readGTSRB,readFood101
from dataProcess import preprocessingCIFAR, preprocessingMINST, preprocessingflower
from train import getTrainModelData
from dataProcess import load_data , shuffleAndSplitData , clipDataTopX


def initializeData(dataset , shouldPreprocess ,dataFolderPath):
    if dataset == 'cifar10':
        print("Loading cifar10 data")
        ## 读取正常的 CIFAR10 数据集
        if (shouldPreprocess):
            dataX, dataY, _, _ = readCIFAR10()
        else:
            # 读取特征提取以后的训练数据
            dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)
        # 将数据集分成 4 部分，每个 10520 条数据
        cluster = 10520
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            dataX, dataY, cluster)

        # 对数据进行预处理
        if (shouldPreprocess):
            toTrainDataSave, toTestDataSave = preprocessingCIFAR(toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)
        else:
            # 特征提取过的数据不需要再预处理了
            toTrainDataSave, toTestDataSave = toTrainData, toTestData
            shadowDataSave, shadowTestDataSave = shadowData, shadowTestData
        print(toTrainDataSave.shape)

    if dataset == 'mnist':
        print("Loading mnist data")

        if (shouldPreprocess):
            #读取正常的 mnist 数据集
            print("读取正常的 mnist 数据集")
            dataX, dataY, _, _ = readMINST()
        else:
            # 读取特征提取以后的训练数据
            dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)

        # #将数据集分成 4 部分，每个 10520 条数据
        cluster = 10520
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            dataX, dataY, cluster)

        # # 对数据进行预处理
        if (shouldPreprocess):
            toTrainDataSave, toTestDataSave = preprocessingMINST(toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = preprocessingMINST(shadowData, shadowTestData)
        else:
            # 特征提取过的数据不需要再预处理了
            toTrainDataSave, toTestDataSave = toTrainData, toTestData
            shadowDataSave, shadowTestDataSave = shadowData, shadowTestData
        print(toTrainDataSave.shape)

    if dataset == 'fashionmnist':
        print("Loading fashionmnist data")

        if (shouldPreprocess):
            # 读取正常的 fashionmnist 数据集
            print("读取正常的 fashionmnist 数据集")
            dataX, dataY, _, _ = readMINST()
        else:
            # 读取特征提取以后的训练数据
            dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)

        # #将数据集分成 4 部分，每个 10520 条数据
        cluster = 10520
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            dataX, dataY, cluster)

        # # 对数据进行预处理
        if (shouldPreprocess):
            toTrainDataSave, toTestDataSave = preprocessingMINST(toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = preprocessingMINST(shadowData, shadowTestData)
        else:
            # 特征提取过的数据不需要再预处理了
            toTrainDataSave, toTestDataSave = toTrainData, toTestData
            shadowDataSave, shadowTestDataSave = shadowData, shadowTestData
        print(toTrainDataSave.shape)

    if dataset == 'cifar100':

        print("Loading CIFAR100 data")

        #dataX, dataY, _, _ = readCIFAR100()
        # 读取特征提取以后的训练数据
        dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)
        # #将数据集分成 4 部分，每个 10520 条数据
        cluster = 10520
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            dataX, dataY, cluster)

        print(shadowData.shape, shadowTestData.shape)
        # # 对数据进行预处理
        if (shouldPreprocess):
            toTrainDataSave, toTestDataSave = preprocessingCIFAR(toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)
        else:
            # 数据不需要再预处理了
            toTrainDataSave, toTestDataSave = toTrainData, toTestData
            shadowDataSave, shadowTestDataSave = shadowData, shadowTestData
        print(toTrainDataSave.shape)

    if dataset == 'svhn':  #(10520, 3, 32, 32)
        print("Loading svhn data")
        # 读取正常的 svhn 数据集
        # dataX, dataY, _, _ = readSVHN()
        # 读取特征提取以后的标签
        dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)
        # 将数据集分成 4 部分，每个 10520 条数据
        cluster = 10520
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            dataX, dataY, cluster)

        # 对数据进行预处理
        if (shouldPreprocess):
            toTrainDataSave, toTestDataSave = preprocessingCIFAR(toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)
        else:
            # 特征提取过的数据不需要再预处理了
            toTrainDataSave, toTestDataSave = toTrainData, toTestData
            shadowDataSave, shadowTestDataSave = shadowData, shadowTestData
        print(toTrainDataSave.shape)

    if dataset == 'lfw':
        print("Loading lfw data")
        dataX, dataY, testX, testY = readLFW()
        # 读取特征提取以后的标签
        # dataX, dataY = readFeatureExtractedCIFAR10()
        print("Preprocessing data")
        print("------", dataX.shape)  #------ (1680, 1, 125, 94)
        cluster = 420   # #将数据集分成 4 部分，每个 420 条数据
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

        print("=====", toTrainData.shape, shadowTestData.shape)  # ===== (420, 1, 125, 94) (420, 1, 125, 94)

    if dataset == 'stl':
        print("Loading STL data")
        ## 读取正常的 STL数据集
        # dataX, dataY, _, _ = readSTL()
        # 读取特征提取以后的标签
        dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)  # (5000,) (5000, 3, 96, 96)
        # 将数据集分成 4 部分，每个 10520 条数据
        cluster = 1250
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
            shuffleAndSplitData(dataX, dataY, cluster)

        toTrainDataSave, toTestDataSave, shadowDataSave, shadowTestDataSave = \
            toTrainData, toTestData, shadowData, shadowTestData

        print("=====", toTrainData.shape, shadowTestData.shape)  # ===== (420, 1, 125, 94) (420, 1, 125, 94)

    if dataset == 'gtsrb':
        print("Loading GTSRB data")
        if (shouldPreprocess):
            # 读取正常的 GTSRB 数据集
            print("读取正常的 GTSRB 数据集")
            dataX, dataY, _, _ = readGTSRB()
        else:
            # 读取特征提取以后的训练数据
            dataX, dataY = readFeatureExtractedCIFAR10()
        print(dataY.shape, dataX.shape)
        # #将数据集分成 4 部分，(26640, 3, 32, 32)
        cluster = 6660
        toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            dataX, dataY, cluster)

        # 对数据进行预处理
        if (shouldPreprocess): # (6000, 3, 32, 32) 43
            toTrainDataSave, toTestDataSave = preprocessingflower(toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = preprocessingflower(shadowData, shadowTestData)
        else:
            # 特征提取过的数据不需要再预处理了
            toTrainDataSave, toTestDataSave = toTrainData, toTestData
            shadowDataSave, shadowTestDataSave = shadowData, shadowTestData
        print(toTrainDataSave.shape)

    print(toTrainDataSave.shape)
    # dataPath =  './data/cifar10/Preprocessed' ## 设置存储的路径
    # dataPath = './data/mnist/Preprocessed'  ## 设置存储的路径
    # dataPath = './data/fashionmnist/Preprocessed'  ## 设置存储的路径
    # dataPath = './data/cifar100/Preprocessed'  ## 设置存储的路径
    # dataPath = './data/lfw/Preprocessed'  ## 设置存储的路径
    # dataPath = './data/gtsrb/Preprocessed'  ## 设置存储的路径
    dataPath = dataFolderPath + '/Preprocessed'
    #将处理好的数据进行保存    
    np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
    np.savez(dataPath + '/targetTest.npz', toTestDataSave, toTestLabel)
    np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
    np.savez(dataPath + '/shadowTest.npz', shadowTestDataSave, shadowTestLabel)

    print("数据预处理结束。。。Preprocessing finished\n\n")



def initializeTargetModel(attackerModelDataPath , classifierType , num_epoch,dataFolderPath):
    # getTrainModelData()得到最终目标模型的置信度分数和标签数据  用于训练攻击模型
    print("Training the Target model for {} epoch".format(num_epoch))

    # cifar10
    # targetTrain, targetTrainLabel = load_data( './data/cifar10/Preprocessed/targetTrain.npz')
    # targetTest, targetTestLabel = load_data( './data/cifar10/Preprocessed/targetTest.npz')

    # MNIST
    # targetTrain, targetTrainLabel = load_data('./data/mnist/Preprocessed/targetTrain.npz')
    # targetTest, targetTestLabel = load_data('./data/mnist/Preprocessed/targetTest.npz')

    # fashionmnist
    # targetTrain, targetTrainLabel = load_data('./data/fashionmnist/Preprocessed/targetTrain.npz')
    # targetTest, targetTestLabel = load_data('./data/fashionmnist/Preprocessed/targetTest.npz')

    # # cifar100
    # targetTrain, targetTrainLabel = load_data('./data/cifar100/Preprocessed/targetTrain.npz')
    # targetTest, targetTestLabel = load_data('./data/cifar100/Preprocessed/targetTest.npz')

    # lfw
    # targetTrain, targetTrainLabel = load_data('./data/lfw/Preprocessed/targetTrain.npz')
    # targetTest, targetTestLabel = load_data('./data/lfw/Preprocessed/targetTest.npz')

    # GTSRB
    # targetTrain, targetTrainLabel = load_data( './data/gtsrb/Preprocessed/targetTrain.npz')
    # targetTest, targetTestLabel = load_data( './data/gtsrb/Preprocessed/targetTest.npz')

    targetTrain, targetTrainLabel = load_data(dataFolderPath + '/Preprocessed/targetTrain.npz')
    targetTest, targetTestLabel = load_data(dataFolderPath + '/Preprocessed/targetTest.npz')
    attackModelDataTarget, attackModelLabelsTarget  = getTrainModelData(classifierType,
                                                                        targetTrain,
                                                                        targetTrainLabel,
                                                                        targetTest,
                                                                        targetTestLabel,
                                                                        num_epoch)

    np.savez(attackerModelDataPath + '/targetModelData.npz', attackModelDataTarget, attackModelLabelsTarget)
    return attackModelDataTarget, attackModelLabelsTarget


def initializeShadowModel(attackerModelDataPath , classifierType,num_epoch , dataFolderPath ):
    # getTrainModelData()得到最终目标模型的置信度分数和标签数据  用于攻击模型
    print("Training the Shadow model for {} epoch".format(num_epoch))
    # cifar10
    # shadowTrainData, shadowTrainLabel = load_data('./data/cifar10/Preprocessed/shadowTrain.npz')
    # shadowTestData, shadowTestLabel = load_data('./data/cifar10/Preprocessed/shadowTest.npz')

    # MNIST
    # shadowTrainData, shadowTrainLabel = load_data('./data/mnist/Preprocessed/shadowTrain.npz')
    # shadowTestData, shadowTestLabel = load_data('./data/mnist/Preprocessed/shadowTest.npz')


    # fashionmnist
    # shadowTrainData, shadowTrainLabel = load_data('./data/fashionmnist/Preprocessed/shadowTrain.npz')
    # shadowTestData, shadowTestLabel = load_data('./data/fashionmnist/Preprocessed/shadowTest.npz')


    # cifar100
    # shadowTrainData, shadowTrainLabel = load_data('./data/cifar100/Preprocessed/shadowTrain.npz')
    # shadowTestData, shadowTestLabel = load_data('./data/cifar100/Preprocessed/shadowTest.npz')


    # lfw
    # shadowTrainData, shadowTrainLabel = load_data('./data/lfw/Preprocessed/shadowTrain.npz')
    # shadowTestData, shadowTestLabel = load_data('./data/lfw/Preprocessed/shadowTest.npz')


    # gtsrb
    # shadowTrainData, shadowTrainLabel = load_data('./data/gtsrb/Preprocessed/shadowTrain.npz')
    # shadowTestData, shadowTestLabel = load_data('./data/gtsrb/Preprocessed/shadowTest.npz')

    shadowTrainData, shadowTrainLabel = load_data(dataFolderPath + '/Preprocessed/shadowTrain.npz')
    shadowTestData, shadowTestLabel = load_data(dataFolderPath + '/Preprocessed/shadowTest.npz')

    attackModelDataShadow, attackModelLabelsShadow  = getTrainModelData(classifierType,
                                                                        shadowTrainData,
                                                                        shadowTrainLabel,
                                                                        shadowTestData,
                                                                        shadowTestLabel,
                                                                        num_epoch)

    np.savez(attackerModelDataPath + '/shadowModelData.npz', attackModelDataShadow, attackModelLabelsShadow)
    return attackModelDataShadow, attackModelLabelsShadow


def generateAttackData(dataset, classifierType, num_epoch, preprocessData, trainTargetModel, trainShadowModel,
                       dataFolderPath):

    # 存放目标模型数据的查询结果 和 影子模型查询结果的 路径
    # attackerModelDataPath = './data/cifar10/attackerModelData'
    # attackerModelDataPath = './data/mnist/attackerModelData'
    # attackerModelDataPath = './data/fashionmnist/attackerModelData'
    # attackerModelDataPath = './data/cifar100/attackerModelData'
    # attackerModelDataPath = './data/lfw/attackerModelData'
    # attackerModelDataPath = './data/gtsrb/attackerModelData'

    # 存放目标模型数据的查询结果 和 影子模型查询结果的 路径
    attackerModelDataPath = dataFolderPath + '/attackerModelData'

    if (preprocessData):
        # 读取数据，并进行预处理
        # 使用特征提取后的数据训练，这里应该设为 False
        ## 注意！！！CIFAR100 都为False
        shouldPreprocess = True
        initializeData(dataset, shouldPreprocess, dataFolderPath)

    if (trainTargetModel):
        targetX, targetY = initializeTargetModel(attackerModelDataPath, classifierType, num_epoch, dataFolderPath)
    else:
        targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')

    if (trainShadowModel):
        shadowX, shadowY = initializeShadowModel(attackerModelDataPath, classifierType, num_epoch, dataFolderPath)
    else:
        shadowX, shadowY = load_data(attackerModelDataPath + '/shadowModelData.npz')

    # 对查询得到的置信度分数取最大的3个
    targetX = clipDataTopX(targetX, top=3)
    shadowX = clipDataTopX(shadowX, top=3)

    return targetX, targetY, shadowX, shadowY
