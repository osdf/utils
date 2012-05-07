"""
PCA -- principle components
"""


import scipy.linalg as la
import numpy as np


def pca(data, covered=None, whiten=False, retained=0):
    """
    Compute principle components of _data_.
    _data_ has samples per row. It is assumed
    that _data_ is already normalized (often:
    mean of samples is zero). _covered_ determines
    how much variance should be covered, if None,
    full rotation is done. Set _whiten_, if
    _data_ should be white after transformation
    (i.e. empirical covariance is I). 
    
    Use this method, if data dimensionlity (data.shape[1])
    is much smaller than number of samples (data.shape[0]):
    PCA is computed via forming the covariance matrix and
    then doing svd on it.

    Returns:
    - rotated, _data_ rotated into principle subspace
    - comp, principle components (column wise)
    - s, eigenvalues of covariance matrix (only retained ones).
    """
    n, d = data.shape
    # working with covariance + (svd on cov.) is 
    # much faster than svd on data directly.
    cov = np.dot(data.T, data)/n
    u, s, v = la.svd(cov, full_matrices=False)
    if covered is None:
        assert retained > 0, "How many components should be used?\n" \
                "Set _retained_ or _covered_."
    else:
        total = np.cumsum(s)[-1]
        retained = sum(np.cumsum(s/total) <= covered)
    s = s[0:retained]
    u = u[:,0:retained]
    if whiten:
        comp = np.dot(u, np.diag(1./np.sqrt(s)))
    else:
        comp = u
    rotated = np.dot(data, comp)
    return rotated, comp, s


def pca_apply(data, comp):
    """
    apply components _comp_ (from pca above)
    to _data_. Data (samples per row) should
    be accordingly normalized.
    """
    return np.dot(data, comp)


def zca(data, eps=0.1, **schedule):
    """
    Zero Components Analysis:
    - whitening of data
    - eigenvalues of covariance are at least _eps_
    - after whitening, rotate back in original space.
    _data_ has rowwise samples.
    _data_ is assumed to be already normalized
    (usually: mean of samples is zero).

    Returns:
    - zca, the processed _data_
    - comp, the complete transformation matrix (right multiply!)
    - s, eigenvalues of covariance matrix
    """
    n, d = data.shape
    cov = np.dot(data.T, data)/n
    u, s, v = la.svd(cov, full_matrices=False)
    comp = np.dot(np.dot(u, np.diag(1./np.sqrt(s + eps))), u.T)
    zca = np.dot(data, comp)
    return zca, comp, s


def unwhiten(data, comp):
    """
    Inverse process of whitening.

    _data_ is in principle subspace
    coordinates, rowwise. _comp_ (column wise)
    span this principle subspace. 
    Unwhitening needs to be done with
    the pseudo inverse, as comp usually
    is not a full rotation matrix.
    """
    uw = la.pinv2(comp)
    return np.dot(data, uw)
