import pandas as pd
import numpy as np
import itertools as it

# 初始化函数，得到一项集
# IUPUT：数据列表（所有的数据组成的list）
# OUTPUT：一项集，项集类型是frozenset


def init(dataList):
    InitDataSet = set(dataList)
    setList = list(InitDataSet)
    itemList = []

    for i in range(len(setList)):
        itemList.append([setList[i]])

    itemList.sort()
    itemList = list(map(frozenset, itemList))
    return itemList

# 扫描函数，筛选出满足最小支持度的项集
# IUPUT：原始数据，项集列表，不满足最小支持度的项集列表，最小支持度
# OUTPUT：满足最小支持度的项集列表，不满足最小支持度的项集列表，项集支持度字典


def scan(data, itemList, itemDownList, miniSupport):
    # 生成计数字典，记录每个项集在每一次记录中出现的次数
    countDic = {}
    num = data.shape[0]
    for d in data:
        for item in itemList:
            if item.issubset(d):
                countDic[item] = countDic.get(item, 0) + 1

    # 记录满足最小支持度的项集以及不满足的项集，生成满足最小支持度的项集与其支持度的键值对
    itemUpList = []
    supportDic = {}
    for k in countDic:
        support = float(countDic[k] / num)
        if support >= miniSupport:
            itemUpList.append(k)
            supportDic[k] = support
        else:
            itemDownList.append(k)

    return itemUpList, itemDownList, supportDic

# 连接函数，生成更大的项集
# IUPUT：满足最小支持度的项集列表，不满足最小支持度的项集列表，连接后的项集中元素的个数，即连接后为k项集
# OUTPUT：进行连接后的项集列表


def link(itemUpList, itemDownList, k):
    itemListNew = []
    itemListemp = []
    lenLu = len(itemUpList)
    lenLd = len(itemDownList)
    for i in range(lenLu):
        for j in range(i + 1, lenLu):
            La = list(itemUpList[i])[:k - 2]
            Lb = list(itemUpList[j])[:k - 2]
            La.sort()
            Lb.sort()
            if La == Lb:
                itemListemp.append(itemUpList[i] | itemUpList[j])
    if lenLd > 0:
        for i in range(lenLd):
            for j in range(len(itemListemp)):
                if itemDownList[i].issubset(itemListemp[j]) == False:
                    itemListNew.append(itemListemp[j])
        itemListNew = list(set(itemListNew))
        return itemListNew
    else:
        return itemListemp

# 逻辑整合的主函数
# IUPUT：原始数据，数据列表（所有的数据组成的list），最小支持度，
# OUTPUT：进行连接后的项集列表的列表（比如L[0]为1项集列表），最大项集的元素个数（即最大为几项集），满足最小支持度的项集/支持度的键/值对


def apriori(data, dataList, miniSupport):
    itemList = init(dataList)
    itemDownList = []
    itemUpList, itemDownList, supportDataDic = scan(
        data, itemList, itemDownList, miniSupport)
    L = [itemUpList]
    k = 2
    while(len(L[k - 2]) > 0):
        itemList = link(L[k - 2], itemDownList, k)
        itemUpList, itemDownList, supportDic = scan(
            data, itemList, itemDownList, miniSupport)
        supportDataDic.update(supportDic)
        L.append(itemUpList)
        k += 1
    return L, k - 2, supportDataDic
