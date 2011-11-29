"""

"""


import numpy as np
import scipy.linalg as la
from scipy.optimize import fmin_l_bfgs_b, fmin_tnc


def check_grad(f, fprime, x0, args, eps=10**-4, verbose=False):
    """
    """
    # computed gradient at x0
    grad = fprime(x0, **args)
    # space for the numeric gradient
    ngrad = np.zeros(grad.shape)
    # for every component of x:
    if verbose: 
        print "Number of total function calls: 2*%d"% x0.shape[0]
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
    delta_2 = np.sum((grad-ngrad)**2)
    if verbose:
        print "Squared distance: %f"% delta_2
    return np.sqrt(delta_2)


def msgd(func, x0, fprime, args, batch_args,
        epochs, nos, lr, btsz, verbose=False, 
        **params):
    """
    Minibatch stochastic gradient descent.
    """
    div = nos/btsz
    mod = nos%btsz
    lr /= btsz
    scores = []
    for e in xrange(epochs):
        sc = 0
        for b in xrange(div):
            for item in batch_args:
                args[item] = batch_args[item][b*btsz:(b+1)*btsz]
            sc += func(x0, **args)
            delta = fprime(x0, **args)
            x0 -= lr*delta
        if mod>0:
            for item in batch_args:
                args[item] = batch_args[item][b*btsz:(b+1)*btsz]
            sc += func(x0, **args)
            delta = fprime(x0, **args)
            x0 -= lr*btsz*x0/mod
        scores.append(sc)
        if verbose:
            print sc
    lr *= btsz
    return x0, scores


def lbfgsb(func, x0, fprime=None, args=(), approx_grad=0, 
        bounds=None, m=10, factr=10000000.0, pgtol=1e-05, 
        epsilon=1e-08, iprint=-1, maxfun=15000, disp=2, **params):
    """
    """
    return fmin_l_bfgs_b(func=func, x0=x0, fprime=fprime, 
            args=args, approx_grad=approx_grad, bounds=bounds, 
            m=m, factr=factr, pgtol=pgtol, epsilon=epsilon, 
            iprint=iprint, maxfun=maxfun, disp=disp)


def tnc(func, x0, fprime=None, args=(), approx_grad=0, bounds=None,
        epsilon=1e-08, scale=None, offset=None, messages=15, 
        maxCGit=-1, maxfun=None, eta=-1, stepmx=0, accuracy=0, fmin=0, 
        ftol=-1, xtol=-1, pgtol=-1, rescale=-1, disp=5, **params):
    """
    """
    return fmin_tnc(func=func, x0=x0, fprime=fprime, args=args, 
            approx_grad=approx_grad, bounds=bounds, epsilon=epsilon, 
            scale=scale, offset=offset, messages=messages, 
            maxCGit=maxCGit, maxfun=maxfun, eta=eta, stepmx=stepmx, 
            accuracy=accuracy, fmin=fmin, ftol=ftol, xtol=xtol, 
            pgtol=pgtol, rescale=rescale, disp=disp)
