"""
Marginalized Denoising Autoencoder
"""


import numpy as np
from scipy.linalg import lstsq


def mDA(X, p):
    """
    _X_ is the data in 'rows', with an
    added bias entry (equal to 1)! _p_
    is the denoising probability.
    """
    n, d = X.shape
    
    q = np.ones(1, d) * (1-p)
    q[0, 0] = 1
    
    S = np.dot(X.T, X)
    Q = S*np.dot(q.T, q)
    np.fill_diagonal(Q, q*np.diag(S))
    P = S*q
    # Now that we have a lstsq formulation,
    # consider adding bayesian things in?
    # compare to lr.py?
    wm = lstsq((Q+1e-5*np.eye(d)).T, P.T[:, 1:])[0]
    # note: first _row_ of wm has bias values
    return wm

