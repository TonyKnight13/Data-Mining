import  numpy as np

def standard_deviation_normalization(data):
    '''
    标准差正规化
    '''
    m,n = data.shape
    datanew = np.zeros((m,n))
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0)
    
    datanew = (data - data_mean) / data_std
    return datanew

