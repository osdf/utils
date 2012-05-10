"""
Canoncial Correlation Analysis.
"""


import numpy as np
import scipy.linalg as la


def cca(X, Y, k, SMALL=1e-5):
    """Standard CCA.

    Views _X_ and _Y_ are per *row*.

    For an explanation of the algorithm, 
    see Section 6.4 and 6.5 in
    Kernel Methods for Pattern Analysis.
    """
    n, dx = X.shape
    C = np.cov(X.T, Y.T)
    Cxy = C[:dx, dx:]
    Cxx = C[:dx, :dx] + SMALL * np.eye(dx)
    Cyy = C[dx:, dx:] + SMALL * np.eye(Y.shape[1])

    # Do not use la.sqrtm.
    # This can be done by hand...
    xeval, xevec = la.eigh(Cxx)
    yeval, yevec = la.eigh(Cyy)

    # ... because the inverses are then simple
    isqrtx = np.dot(xevec, (xevec/np.sqrt(xeval)).T)
    isqrty = np.dot(yevec, (yevec/np.sqrt(yeval)).T)

    tmp = np.dot(isqrtx, Cxy)
    tmp = np.dot(tmp, isqrty)
    [U, S, V] = la.svd(tmp, full_matrices=False)

    ccX = np.dot(isqrtx, U[:,:k])
    ccY = np.dot(isqrty, V[:k].T)
    
    return ccX, ccY


def pccA(X, Y, k, SMALL=1e-5):
    """Probabilistic CCA.

    See A probabilistic interpretation of canonical correlation analysis
    and/or Robust probabilistic projections.
    """
    pass


def rccA(X, Y, k):
    """Robust CCA.

    See Robust probabilistic projections.
    """
    pass
