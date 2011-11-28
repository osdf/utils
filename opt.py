"""

"""


import numpy as np
import scipy.linalg as la


def check_grad(f, fprime, x0, args, eps=10**-4):
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


def sgd(func, x0, fprime, args, 
        epochs, lr, btsz, verbose):
#weights, inputs, targets, epochs, lr, btsz, lmbd):
    """
    Stochastic gradient descent.
    """
    n, _ = inputs.shape
    div = n/btsz
    mod = n%btsz
    lr /= btsz
    scores = []
    for e in xrange(epochs):
        sc = 0
        for b in xrange(div):
            sc += func(x0, inputs[b*btsz:(b+1)*btsz],
                    targets[b*btsz:(b+1)*btsz], lmbd)
            delta = fprime(weights, inputs[b*btsz:(b+1)*btsz], 
                    targets[b*btsz:(b+1)*btsz], lmbd)
            x0 -= lr*delta
        if mod>0:
            sc += score(x0, inputs[-mod:],targets[-mod:], lmbd)
            delta = grad(x0, inputs[-mod:], targets[-mod:], lmbd)
            x0 -= lr*btsz*x0/mod
        scores.append(sc)
    lr *= btsz
    return scores
