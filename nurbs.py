import numpy as np


def obj_mat_to_3D(Pw):
    ''' Convert a 4D (homogeneous) object matrix to a 3D object matrix,
    i.e. a (... x 4) to a (... x 3) matrix. '''
    Pw = np.asfarray(Pw)
    if Pw.shape[-1] == 3:
        return Pw
    w = Pw[...,-1]
    return Pw[...,:-1] / w[...,np.newaxis]


def obj_mat_to_4D(P, w=None):
    ''' Idem obj_mat_to_3D, vice versa.  If w is None, all weights are
    set to unity, otherwise it is assumed that w has one less dimension
    than P. '''
    P = np.asfarray(P)
    s = P.shape
    if s[-1] == 4:
        return P
    Pw = np.ones(list(s[:-1]) + [4])
    Pw[...,:-1] = P
    if w is not None:
        Pw *= w[...,np.newaxis]
    return Pw
