import pandas as pd
import numpy as np


def cauDis(data, datai):
    m, n = data.shape
    sumDis = np.sum((data - datai) ** 2, axis=1)
    sortedDisIndex = np.argsort(sumDis)
    sumDis.sort()
    return sortedDisIndex, sumDis


def classify(data, datai, dataclass, k):
    sortedDisIndex, sumDis = cauDis(data, datai)
    countDic = {}
    for i in range(k):
        Label = frozenset(dataclass[sortedDisIndex[i]])
        countDic[Label] = countDic.get(Label, 0) + 1

    return max(countDic, key=countDic.get)


def predict(data, dataclass, dataT, k):
    m = dataT.shape[0]
    classes = []
    for d in dataT:
        classes.append(classify(data, d, dataclass, k))
    return classes
