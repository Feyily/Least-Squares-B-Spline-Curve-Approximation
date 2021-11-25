import numpy as np
NDEC = 5


def chord_length_param(n, Q):
    ''' 
    用弦长表示参数的取值（假设u的范围是[0, 1]）
    '''
    Ub = np.zeros(n + 1)
    clk = np.zeros(n + 1)
    for k in range(1, n + 1):
        clk[k] = np.linalg.norm(Q[k] - Q[k-1])
    d = np.sum(clk)
    for k in range(1, n):
        Ub[k] = Ub[k-1] + clk[k] / d
    Ub[n] = 1.0
    return Ub


def approximating_knot_vec_end(n, p, r, k, l, Ub):
    ''' 
    构造结点向量
    '''
    U = np.zeros(n + p + 2)
    for i in range(p + 1):
        U[i], U[n+i+1] = Ub[0], Ub[r]
    nc = n - k - l
    w = np.zeros(nc + 1)
    inc = float(r + 1) / float(nc + 1)
    low = high = 0
    d = -1
    for i in range(nc + 1):
        d += inc
        high = int(np.floor(d + 0.5))
        sum = 0.0
        for j in range(low, high + 1):
            sum += Ub[j]
        w[i] = sum / (high - low + 1)
        low = high + 1
    it = 1 - k
    ie = nc - p + l
    r = p
    for i in range(it, ie + 1):
        js = max(0, i)
        je = min(nc, i + p - 1)
        r += 1
        sum = 0
        for j in range(js, je + 1):
            sum += w[j]
        U[r] = sum / (je - js + 1)
    clean_knot_vec(U)
    return U


def clean_knot_vec(U):
    ''' 
    Clean the entire knot vector U, i.e. ensure there are no close
    knots.  For example, this is called automatically by a NURBSObject
    upon instantiation (IN-PLACE). 
    '''
    Ui, ind = np.unique(U.round(decimals=NDEC), return_inverse=True)
    U[:] = Ui[ind]


def check_knot_v(U, u):
    # 检查节点向量的合理性
    u = np.asfarray(u)
    ur = np.round(u, decimals=NDEC)
    u[U[0] == ur] = U[0]
    u[U[-1] == ur] = U[-1]
    if (u < U[0]).any() or (u > U[-1]).any():
        raise Exception(U, u)
    return u


def uni_knot_vec(n, p):
    ''' 
    构造一个均匀的节点向量
    '''
    U = np.zeros(n + p + 2)
    for j in range(1, n - p + 1):
        U[j+p] = float(j) / (n - p + 1)
    U[-p-1:] = 1.0
    clean_knot_vec(U)
    return U
