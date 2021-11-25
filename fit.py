import numpy as np
import scipy.sparse.linalg
import nurbs
import knot
import basis


def global_curve_approx_fixedn_ders(r, Q, p, n, k, Ds, l, De, uk=None):

    ''' 
    给出曲线在起始位置的导数Ds， 和末尾处的导数De，拟合曲线

               Ds^1,...,Ds^k    De^1,...,De^l    k,l < p + 1

    其中 Ds^i 表示曲线在起始位置处的i阶导数， De^j 表示曲线在末尾处的j阶导数.

    Parameters
    ----------
    r + 1 = 待拟合散点的个数
    Q = 矩阵形式的散点
    p = 拟合曲线使用的阶数
    n + 1 = 拟合使用的控制顶点数
    k, l = 待拟合曲线起始位置、末尾导数的个数
    Ds, De = 待拟合曲线起始位置、末尾导数
    uk = 每个散点对应的参数值（可以置空）

    Returns
    -------
    U, Pw = 拟合出的曲线的结点向量和控制顶点

    Source
    ------
    Piegl & Tiller, Least-Squares B-spline Curve Approximation with
    Arbitrary End Derivatives, Engineering with Computers, 2000.

    '''

    if p < 1 or n < p or p < k or p < l or n < k + l + 1:
        raise Exception(p, n, k, l)
    if not r > n:
        raise Exception(r, n)
    Q = nurbs.obj_mat_to_3D(Q)
    Ds, De = [np.asfarray(V) for V in (Ds, De)]
    if uk is None:
        uk = knot.chord_length_param(r, Q)
    U = knot.approximating_knot_vec_end(n, p, r, k, l, uk)
    P = np.zeros((n + 1, 3))
    P[0], P[n] = Q[0], Q[r]
    if k > 0:
        ders = basis.ders_basis_funs(p, uk[0], p, k, U)
        for i in range(1, k + 1):
            Pi = Ds[i-1]
            for h in range(i):
                Pi -= ders[i,h] * P[h]
            P[i] = Pi / ders[i,i]
    if l > 0:
        ders = basis.ders_basis_funs(n, uk[-1], p, l, U)
        for j in range(1, l + 1):
            Pj = De[j-1]
            for h in range(j):
                Pj -= ders[j,-h-1] * P[-h-1]
            P[-j-1] = Pj / ders[j,-j-1]
    if n > k + l + 1:
        Ps = [P[   i][:,np.newaxis] for i in range(k + 1)]
        Pe = [P[-j-1][:,np.newaxis] for j in range(l + 1)]
        lu, NT, Os, Oe = build_decompose_NTN(r, p, n, uk, U, k, l)
        R = Q.copy()
        for i in range(k + 1):
            R -= (Os[i] * Ps[i]).T
        for j in range(l + 1):
            R -= (Oe[j] * Pe[j]).T
        for i in range(3):
            rhs = np.dot(NT, R[1:-1,i])
            P[k+1:-l-1,i] = lu.solve(rhs)
    Pw = nurbs.obj_mat_to_4D(P)
    return U, Pw


def build_decompose_NTN(r, p, n, uk, U, k=0, l=0):

    ''' 
    构造NTN矩阵，并返回该矩阵的lu分解（见论文）
    '''

    m = n + p + 1
    Os = [basis.one_basis_fun_v(p, m, U, i, uk, r + 1)
          for i in range(k + 1)]
    Oe = [basis.one_basis_fun_v(p, m, U, n - j, uk, r + 1)
          for j in range(l + 1)]
    if k + l + 1 == n:
        return None, None, Os, Oe
    N = np.zeros((r + 1, n + 1))
    spans = basis.find_span_v(n, p, U, uk, r + 1)
    bfuns = basis.basis_funs_v(spans, uk, p, U, r + 1)
    spans0, spans1 = spans - p, spans + 1
    for s in range(r + 1):
        N[s,spans0[s]:spans1[s]] = bfuns[:,s]
    N = N[1:-1,k+1:-l-1]
    NT = N.transpose()
    NTN = np.dot(NT, N)
    NTN = scipy.sparse.csc_matrix(NTN)
    lu = scipy.sparse.linalg.splu(NTN)
    return lu, NT, Os, Oe
