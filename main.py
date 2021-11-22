"""
Forked from https://github.com/CompAero/Genair
"""

import math
import numpy as np

from fit import *


def genrate_points():
    # 螺旋线
    t = np.linspace(0, 10, 100)
    x = 5 * math.cos(t * (5 * 360))
    y = 5 * math.sin(t * (5 * 360))
    z = 10 * t


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
