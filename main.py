"""
Forked from https://github.com/CompAero/Genair
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit import *


def construct_flat_grid(Us, nums=None):
    '''
    生成采样点
    '''
    Us = [np.asfarray(U) for U in Us]
    if nums is not None:
        Us = [np.linspace(U[0], U[-1], num)
              for U, num in zip(Us, nums)]
    if len(Us) == 1:
        return Us
    if len(Us) == 2:
        U, V = Us
        numu, numv = len(U), len(V)
        us = U[:, np.newaxis]
        vs = V[np.newaxis, :]
        us = us.repeat(numv, axis=1)
        vs = vs.repeat(numu, axis=0)
        return [u.flatten() for u in (us, vs)]
    if len(Us) == 3:
        U, V, W = Us
        numu, numv, numw = len(U), len(V), len(W)
        us = U[:, np.newaxis, np.newaxis]
        vs = V[np.newaxis, :, np.newaxis]
        ws = W[np.newaxis, np.newaxis, :]
        us = us.repeat(numv, axis=1).repeat(numw, axis=2)
        vs = vs.repeat(numu, axis=0).repeat(numw, axis=2)
        ws = ws.repeat(numu, axis=0).repeat(numv, axis=1)
        return [u.flatten() for u in (us, vs, ws)]


def rat_curve_point_v(n, p, U, Pw, u, num):
    """ 
    计算曲线在参数u处的函数值
    """

    u = np.asfarray(u)
    Cw = np.zeros((4, num))
    span = basis.find_span_v(n, p, U, u, num)
    N = basis.basis_funs_v(span, u, p, U, num)
    for j in range(p + 1):
        Cw += N[j] * Pw[span - p + j].T
    return Cw[:3] / Cw[-1]


def genrate_points():
    # 螺旋线
    t = np.linspace(0, 1, 1000)
    _x = np.cos(t * 5 * np.pi)
    _y = 5 * np.exp(t)
    _z = t
    ax.scatter(_x, _y, _z)

    Ds = np.array([[
        (5 * np.pi) * np.sin(t[0] * 5 * np.pi),
        5 * np.exp(t[0]),
        1
    ], ])
    De = np.array([[
        (5 * np.pi) * np.sin(t[-1] * 5 * np.pi),
        5 * np.exp(t[-1]),
        1
    ], ])
    return _x, _y, _z, Ds, De


def fit_curve(Q, Ds, De, ncp, p=3):
    '''
    Parameters
    ----------
    Q = 要拟合的散点
    Ds = 曲线头部的导数
    De = 曲线尾部的导数
    ncp = 拟合使用的控制顶点数
    p = 拟合曲线使用的阶数

    Returns
    -------
    U = 拟合出曲线的结点向量
    Pw = 拟合出曲线的控制顶点
    '''

    U, Pw = global_curve_approx_fixedn_ders(len(Q) - 1, Q, p, ncp - 1,
                                            len(Ds), Ds, len(De), De)
    us, = construct_flat_grid((U,), (1000,))
    n = np.array(Pw.shape[:-1]) - 1
    sp = rat_curve_point_v(n, p, U, Pw, us, len(us))

    ax.scatter(sp[0], sp[1], sp[2])
    plt.show()
    return U, Pw


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y, z, ds, de = genrate_points()
    points = np.array([x, y, z]).T
    fit_curve(points, ds, de, 10)
