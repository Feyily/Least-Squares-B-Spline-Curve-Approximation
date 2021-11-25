import numpy as np
import knot


def ders_basis_funs(i, u, p, n, U):

    ''' 
    计算所有非零基函数及其导数（包含n阶导数），返回是一个二维数组`der`，其中
    `ders[k,j]`是N_(i-p+j,p)(u) (0 <= k <= n) 且 (0 <= j <= p)的k阶导数
    '''

    ders = np.zeros((n + 1, p + 1))
    ndu = np.zeros((p + 1, p + 1))
    a = np.zeros((2, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    ndu[0,0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(j):
            ndu[j,r] = right[r+1] + left[j-r]
            tmp = ndu[r,j-1] / ndu[j,r]
            ndu[r,j] = saved + right[r+1] * tmp
            saved = left[j-r] * tmp
        ndu[j,j] = saved
    for j in range(p + 1):
        ders[0,j] = ndu[j,p]
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0,0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2,0] = a[s1,0] / ndu[pk+1,rk]
                d = a[s2,0] * ndu[rk,pk]
            if rk >= - 1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in range(j1, j2 + 1):
                a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1,rk+j]
                d += a[s2,j] * ndu[rk+j,pk]
            if r <= pk:
                a[s2,k] = - a[s1,k-1] / ndu[pk+1,r]
                d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j = s1; s1 = s2; s2 = j
    r = p
    for k in range(1, n + 1):
        for j in range(p + 1):
            ders[k,j] *= r
        r *= p - k
    return ders


def one_basis_fun_v(p, m, U, i, u, num):
    '''
    计算单个基函数，例如N_(i,p)(u)
    '''

    u = np.asfarray(u)
    N = np.zeros((p + 1, num))
    cs = np.ones(num, dtype='bool')
    if i == 0:
        c = u == U[0]
        N[0,c] = 1.0; cs[c] = False
    if i == m - p - 1:
        c = u == U[m]
        N[0,c] = 1.0; cs[c] = False
    c = u < U[i]
    N[0,cs & c] = 0.0; cs[c] = False
    c = u >= U[i+p+1]
    N[0,cs & c] = 0.0; cs[c] = False
    NN = N[:,cs]
    u = u[cs]
    for j in range(p + 1):
        NN[j,(u >= U[i+j]) & (u < U[i+j+1])] = 1.0
    for k in range(1, p + 1):
        if (U[i+k] - U[i]) != 0.0:
            divs = ((u - U[i]) * NN[0]) / (U[i+k] - U[i])
        else:
            divs = None
        saved = np.where(NN[0] == 0.0, 0.0, divs)
        for j in range(p - k + 1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            zs = NN[j+1] == 0.0; nzs = ~ zs
            NN[j] = np.where(zs, saved, NN[j])
            saved = np.where(zs, 0.0, saved)
            if Uright - Uleft != 0.0:
                divs = NN[j+1] / (Uright - Uleft)
            else:
                divs = None
            tmp = np.where(nzs, divs, 0.0)
            NN[j] = np.where(nzs, saved + (Uright - u) * tmp, NN[j])
            saved = np.where(nzs, (u - Uleft) * tmp, saved)
    N[:,cs] = NN
    return N[0]


def find_span_v(n, p, U, u, num):

    ''' 
    查找参数u所在的节点向量区间
    '''

    u = knot.check_knot_v(U, u)
    low = np.array(p).repeat(num)
    high = np.array(n + 1).repeat(num)
    mid = (low + high) // 2
    mid[u == U[n+1]] = n
    c1, c2 = u < U[mid], u >= U[mid+1]
    c3 = u != U[n+1]
    while c1.any() or (c2 & c3).any():
        high = np.where(c1, mid, high)
        low = np.where(c2, mid, low)
        mid = np.where(c1 | c2, (low + high) // 2, mid)
        c1, c2 = u < U[mid], u >= U[mid+1]
    return mid


def basis_funs_v(i, u, p, U, num):

    ''' 
    计算参数u处的基函数
    '''

    N = np.zeros((p + 1, num))
    left = np.zeros((p + 1, num))
    right = np.zeros((p + 1, num))
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j], right[j] = u - U[i+1-j], U[i+j] - u
        saved = 0.0
        for r in range(j):
            tmp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * tmp
            saved = left[j-r] * tmp
        N[j] = saved
    return N
