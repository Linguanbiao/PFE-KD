import argparse
import numpy as np
import sys
from initialization import generateAttackData
import torch
from train import train_model
import warnings
warnings.filterwarnings("ignore")

seed = 21312
sys.dont_write_bytecode = True
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--adv',  default= 1, help='Which adversary 1, 2, or 3')
parser.add_argument('--dataset', default='lfw',
                    help='Which dataset to use (cifar10,cifar100,mnist,lfw,,fashionmnist,gtsrb)')
parser.add_argument('--dataset2', default='111111111111111111',
                    help='Which second dataset for adversary 2 (cifar10,cifar100,mnist,lfw,,fashionmnist,gtsrb)')
parser.add_argument('--classifierType', default='cnn',
                    help='Which classifier cnn or nn')
parser.add_argument('--num_epoch', type=int, default=50,
                    help='Number of epochs to train shadow/target models')
parser.add_argument('--dataFolderPath', default='./data/lfw',
                    help='Path to store dataset1')
parser.add_argument('--dataFolderPath2', default='222222222222222222222',
                    help='Path to store dataset2')
parser.add_argument('--preprocessData', default=True, action='store_true',
                    help='Preprocess the data, if false then load preprocessed data')
parser.add_argument('--trainTargetModel', default=True, action='store_true',
                    help='Train a target model, if false then load an already trained model')
parser.add_argument('--trainShadowModel', default=True, action='store_true',
                    help='Train a shadow model, if false then load an already trained model')
opt = parser.parse_args()


def attackerOne(dataset=opt.dataset,
                classifierType=opt.classifierType,
                num_epoch=opt.num_epoch,
                preprocessData=opt.preprocessData,
                trainTargetModel=opt.trainTargetModel,
                trainShadowModel=opt.trainShadowModel,
                dataFolderPath = opt.dataFolderPath
                ):
    # 生成攻击模型的训练数据和测试数据
    targetX, targetY, shadowX, shadowY = generateAttackData(dataset, classifierType, num_epoch, preprocessData,
                                                            trainTargetModel, trainShadowModel, dataFolderPath)

    print("Training the attack model for the first adversary")
    # 训练攻击模型
    attackModel_one = train_model('softmax', targetX, targetY, shadowX, shadowY, num_epoch=20)

def attackertwo(dataset=opt.dataset,
                dataset2=opt.dataset2,
                classifierType=opt.classifierType,
                num_epoch=opt.num_epoch,
                preprocessData=opt.preprocessData,
                trainTargetModel=opt.trainTargetModel,
                trainShadowModel=opt.trainShadowModel,
                dataFolderPath=opt.dataFolderPath,
                dataFolderPath2=opt.dataFolderPath2):
    # 生成攻击模型的训练数据和测试数据,训练数据来自于第一种数据集，测试数据来自于第二种数据集
    targetX, targetY, _, _ = generateAttackData(dataset, classifierType, num_epoch, preprocessData, trainTargetModel,
                                                trainShadowModel, dataFolderPath)

    shadowX, shadowY, _, _ = generateAttackData(dataset2, classifierType, num_epoch, preprocessData, trainTargetModel,
                                                trainShadowModel, dataFolderPath2)

    print("Training the attack model for the second adversary")
    # 训练攻击模型
    attackModel_one = train_model('softmax', targetX, targetY, shadowX, shadowY, num_epoch=20)


if __name__ == "__main__":
    if (opt.adv == 1):
        attackerOne()
    elif (opt.adv == 2):
        attackertwo()
