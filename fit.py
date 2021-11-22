import numpy as np
import scipy.sparse.linalg
import nurbs
import knot
import basis


def global_curve_approx_fixedn_ders(r, Q, p, n, k, Ds, l, De, uk=None):

    ''' Idem to global_curve_approx_fixedn, but, in addition to the data
    points, end derivatives are also given

               Ds^1,...,Ds^k    De^1,...,De^l    k,l < p + 1

    where Ds^i denotes the ith derivative at the start point and De^j
    the jth derivative at the end.

    Parameters
    ----------
    r + 1 = the number of data points to fit
    Q = the point set in object matrix form
    p = the degree of the fit
    n + 1 = the number of control points to use in the fit
    k, l = the number of start and end point derivatives
    Ds, De = the start and end point derivatives
    uk = the parameter values associated to each data point (if available)

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the fitted curve

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

    ''' Build the matrix of scalars NTN necessary to solve the
    least-squares problem for curve and surface approximations.  The
    matrix is also decomposed using sparse LU.

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
