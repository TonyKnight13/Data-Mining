import pandas as pd
import numpy as np


def cauDisEuclidean(data, datat):
    '''
    计算欧氏距离
    '''
    m, n = data.shape
    sumDis = np.sum((data - datat) ** 2, axis=1)
    sortedDisIndex = np.argsort(sumDis)
    return sortedDisIndex


def cauSimPearson(data, datat):
    '''
    计算皮尔逊相关系数
    '''
    datatemp = np.vstack((data, datat))
    m, n = datatemp.shape
    corrcoef = np.corrcoef(datatemp)
    sortedSimIndex = np.argsort(-corrcoef[-1][1:])
    return sortedSimIndex


def knnClassify(data, datat, dataclass, k, disAlg=cauDisEuclidean):
    sortedIndex = disAlg(data, datat)
    countDic = {}
    for i in range(k):
        Label = frozenset(dataclass[sortedIndex[i]])
        countDic[Label] = countDic.get(Label, 0) + 1

    return max(countDic, key=countDic.get)


def knnPredict(xTrain, yTrain, xTest, k, disAlg=cauDisEuclidean):
    m = xTest.shape[0]
    classes = []
    for d in xTest:
        classes.append(knnClassify(xTrain, d, yTrain, k, disAlg=disAlg))
    return classes
