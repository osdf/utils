"""
Things related to distances.
"""


import numpy as np


def euc_dist(X, Y, squared=True):
    """
    Compute distances between
    rows in X and rows in Y.

    See http://blog.smola.org/post/969195661/in-praise-of-the-second-binomial-formula
    """
    if X is Y:
        Xsq = (X**2).sum(axis=1)
        Ysq = Xsq[np.newaxis, :]
        Xsq = Xsq[:, np.newaxis]
    else:
        Xsq = (X**2).sum(axis=1)[:, np.newaxis]
        Ysq = (Y**2).sum(axis=1)[np.newaxis, :]
    distances = Xsq + Ysq - 2 * np.dot(X, Y.T)
    if squared:
        return distances
    else:
        return np.sqrt(distances)
