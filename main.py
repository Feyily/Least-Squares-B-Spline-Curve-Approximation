"""
Forked from https://github.com/CompAero/Genair
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from fit import *


def genrate_points():
    # 螺旋线
    t = np.linspace(0, 1, 1000)
    x = np.cos(t * 5 * np.pi)
    y = 5 * np.log(t)
    z = t
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()

    Ds = np.array([[
        (5 * np.pi) * np.sin(t[0] * 5 * np.pi),
        5 * np.log(t[0]),
        1
    ], ])
    De = np.array([[
        (5 * np.pi) * np.sin(t[-1] * 5 * np.pi),
        5 * np.log(t[-1]),
        1
    ], ])
    pass


def refit_curve(C, ncp, p=3, num=1000):
    ''' Refit an arbitrary Curve with another Curve of arbitrary degree
    by sampling it at equally spaced intervals.  If possible the
    original end derivatives are kept intact.

    Parameters
    ----------
    C = the Curve to refit
    ncp = the number of control points to use in the fit
    p = the degree to use in the fit
    num = the number of points to sample C with

    Returns
    -------
    Curve = the refitted Curve

    '''

    U, Pw = global_curve_approx_fixedn_ders(num - 1, Q, p, ncp - 1,
                                            len(Ds), Ds, len(De), De, us)
    return curve.Curve(curve.ControlPolygon(Pw=Pw), (p,), (U,))


if __name__ == "__main__":
    genrate_points()
