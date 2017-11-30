import pandas as pd
import numpy as np


def KfoldIdx(m, k_fold=10, shuffle=False):
    '''
    一个Generator，生成训练集和测试集的索引，默认为10折，顺序不变
    '''

    idx = np.arange(m)
    if shuffle:                                                #如果shuffle为True，打乱索引顺序
        np.random.shuffle(idx)
    foldSizes = (m // k_fold) * np.ones(k_fold, dtype=int)     #生成一个list，其长度等于k_fold
    foldSizes[:m % k_fold] += 1                                #最终list每个元素即每一折数据量的上限
    current = 0
    for fz in foldSizes:                                       #通过遍历存有每一折上限的list得到训练集索引，测试集索引
        first, end = current, current + fz
        testIdx = list(idx[first:end])
        trainIdx = list(np.concatenate((idx[:first], idx[end:])))
        yield trainIdx, testIdx
        current = end


def leaveOneOutCrossValidationIdx(m, shuffle=False):
    '''
    一个Generator，生成训练集和测试集的索引，默认顺序不变
    '''
    idx = np.arange(m)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(m):
        testIdx = list(idx[i])
        trainIdx = list(np.concatenate(idx[:i], idx[i + 1:]))
        yield trainIdx, testIdx

##################
###   For KNN  ###
##################

###   k_flod  ###


def test(xTrain, xTest, yTrain, yTest, predictAlg, k, disAlg=cauDisEuclidean):
    '''
    评估预测的准确率
    '''
    right = 0
    classes = list(map(list, predictAlg(
        xTrain, yTrain, xTest, k, disAlg=disAlg)))
    for i in range(len(yTest)):
        if classes[i] == yTest[i]:
            right += 1
    return float(right / len(yTest))


def kfoldCrossValidateForknn(m, data, dataLabel, predictAlg, k, k_fold=10, shuffle=False, disAlg=cauDisEuclidean):
    '''
       进行交叉验证后得到训练集，测试集，并进行预测，最后
    得到的是两个list，存放准确率和每个训练集的索引。
    '''
    kf = KfoldIdx(m, k_fold=k_fold, shuffle=shuffle)
    accuracyRates = []
    trainsIdx = []
    for trainIdx, testIdx in kf:
        xTrain, xTest = data[trainIdx], data[testIdx]
        yTrain, yTest = dataLabel[trainIdx], dataLabel[testIdx]
        accuracyRates.append(
            test(xTrain, xTest, yTrain, yTest, predictAlg, k, disAlg=disAlg))
        trainsIdx.append(trainIdx)
    return accuracyRates, trainsIdx


def findBestK(m, data, dataLabel, predictAlg, k_fold=10, shuffle=False, disAlg=cauDisEuclidean):
    '''
       为了给knn算法选择一个最好的K，进行K值的迭代，k的上界下界分别为0和数据集向量的一半，而且k取奇数，
    避免分类时出现有两个或以上的类别相同的情况出现。
       最后得到的是最高的准确率以及得到这个准确率的model的k值，训练集索引。
    '''
    accuracyRatesks = []
    for k in range(1, int(m / 2) + 1, 2):
        accuracyRates, trainsIdx = kfoldCrossValidateForknn(
            m, data, dataLabel, predictAlg, k, k_fold=k_fold, shuffle=shuffle, disAlg=disAlg)
        accuracyRatesks.append(accuracyRates)
    accuracyRatesks = np.array(accuracyRatesks)
    maxeIdxnum = np.argmax(accuracyRatesks)
    maxRate = np.max(accuracyRatesks)
    maxIdx = list([maxeIdxnum // len(accuracyRatesks[0]),
                   maxeIdxnum % len(accuracyRatesks[0])])
    k = 2 * (maxIdx[0] + 1) + 1
    foldNum = maxIdx[1]
    return maxRate, k, trainsIdx[foldNum]

###  END  ###


###  leave_one_ouT  ###


def leaveOneOutCrossValidateForknn(m, data, dataLabel, predictAlg, k, shuffle=False, disAlg=cauDisEuclidean):
    LOO = leaveOneOutCrossValidationIdx(m, shuffle=shuffle)
    accuracyRates = []
    trainsIdx = []
    for trainIdx, testIdx in LOO:
        xTrain, xTest = data[trainIdx], data[testIdx]
        yTrain, yTest = dataLabel[trainIdx], dataLabel[testIdx]
        accuracyRates.append(test(xTrain, xTest, yTrain, yTest, alg, k))
        trainsIdx.append(trainIdx)
    return accuracyRates, trainsIdx


def LOOFindBestK(m, data, dataLabel, predictAlg, shuffle=False, disAlg=cauDisEuclidean):
    accuracyRatesks = []
    for k in range(1, int(m / 2) + 1, 2):
        accuracyRates, trainsIdx = leaveOneOutCrossValidateForknn(
            m, data, dataLabel, predictAlg, k, shuffle=shuffle, disAlg=disAlg)
        accuracyRatesks.append(accuracyRates)
    accuracyRatesks = np.array(accuracyRatesks)
    maxeIdxnum = np.argmax(accuracyRatesks)
    maxRate = np.max(accuracyRatesks)
    maxIdx = list([maxeIdxnum // len(accuracyRatesks[0]),
                   maxeIdxnum % len(accuracyRatesks[0])])
    k = 2 * (maxIdx[0] + 1) + 1
    foldNum = maxIdx[1]
    return maxRate, k, trainsIdx[foldNum]

###  END  ###

##################
###    END     ###
##################


# the failed one：
#
# def kFoldCrossValidationIdx(m, k, shuffle=False):
#     idx = np.arange(m)
#     if shuffle:
#         np.random.shuffle(idx)
#     idxm = []
#     idxM = []
#     y = m // k
#     left = m - y * k
#     x = y * k
#     for i in range(x):
#         idxm.append(idx[i])
#         if (i + 1) % k == 0:
#             idxM.append(idxm)
#             idxm = []
#     for i in range(left):
#         idxM[i].append(idx[i + x])
#     idxM = np.array(idxM)
#     return idxM


# def kFoldCrossValidation(idxM):
#     idxM = np.array(idxM)
#     m = idxM.shape[0]
#     dataTrainIdx = []
#     dataTestIdx = []
#     for i in range(m):
#         temp = idxM
#         dataTestIdx.append(idxM[i])
#         temp = np.delete(temp, i, axis=0)
#         temp = list(it.chain.from_iterable(temp))
#         dataTrainIdx.append(temp)

#     dataTestIdx = np.array(dataTestIdx)
#     dataTrainIdx = np.array(dataTrainIdx)
#     return dataTrainIdx, dataTestIdx
