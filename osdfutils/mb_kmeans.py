"""
Minibatch Kmeans.
"""


import numpy as np
from metric import euc_dist


def minibatch_k_means(X, k, mbsize, iters):
    """
    Minibatch kmeans according to
    Sculley's paper.

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
        The data matrix.

    k: int
        Number of centers.

    mbsize: int
        Size of minibatch.

    iters: int
        Number of iterations.

    Returns
    -------
    centers: array, shape (k, n_features)
        Computed centers.

    inertia: float
        The value of the inertia criterion 
        given _centers_.
    """

    samples = X.shape[0]
    # Initialize centers
    seed = np.argsort(np.random.rand(samples))[:k]
    centers = X[seed]
    v = np.ones(k)
    #
    for i in xrange(iters):
        sample = np.argsort(np.random.rand(samples))[:mbsize]
        M = X[sample]
        d = np.argmin(euc_dist(M, centers, squared=True), axis=1)
        for j in xrange(mbsize):
            c = d[j]
            v[c] += 1
            eta = 1./v[c]
            centers[c] = (1-eta) * centers[c] + eta * M[j]
    #
    distances = euc_dist(X, centers, squared=True)
    closest_center = np.argmin(distances, axis=1)
    inertia_ = distances[xrange(samples), closest_center].sum()
    return centers, inertia_
