import numpy as np


def obj_mat_to_3D(Pw):
    ''' 
    将齐次矩阵（4维）转换成三维矩阵
    '''
    Pw = np.asfarray(Pw)
    if Pw.shape[-1] == 3:
        return Pw
    w = Pw[...,-1]
    return Pw[...,:-1] / w[...,np.newaxis]


def obj_mat_to_4D(P, w=None):
    ''' 
    和obj_mat_to_3D相反，将三维矩阵转换成齐次矩阵。如果w是None，则所有的权重
    均为1；如果给定了w，则使用w的权重（w比P少一维）
    '''
    P = np.asfarray(P)
    s = P.shape
    if s[-1] == 4:
        return P
    Pw = np.ones(list(s[:-1]) + [4])
    Pw[...,:-1] = P
    if w is not None:
        Pw *= w[...,np.newaxis]
    return Pw
