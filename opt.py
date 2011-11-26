"""

"""


import numpy as np
import scipy.linalg as la


def check_grad(f, fprime, x0, eps=10**-4, **args):
    """
    """
    # computed gradient at x0
    grad = fprime(x0, **args)
    # space for the numeric gradient
    ngrad = np.zeros(grad.shape)
    # for every component of x:
    for i in xrange(x0.shape[0]):
        # inplace change
        x0[i] += eps
        f1 = f(x0, **args)
        # inplace change
        x0[i] -= 2*eps
        f2 = f(x0, **args)
        # second order approximation
        ngrad[i] = (f1-f2)/(2*eps)
        # undo previous _inplace_ change 
        x0[i] += eps
    return np.sqrt(np.sum((grad-ngrad)**2))
