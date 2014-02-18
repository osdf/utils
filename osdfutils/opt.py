"""
Some basic optimization facilities.
"""


import numpy as np
import scipy.linalg as la
from scipy.optimize import fmin_l_bfgs_b, fmin_tnc
import sys


def smd(x0, fandprime, args, batch_args,
        epochs, nos, lmbd=0.99, mu=0.02,
        eta0=0.0005, btsz=5, verbose=False,
        **params):
    """
    Stochastic Meta Descent.
    """
    ii = 1j*sys.float_info.min
    p = np.size(x0)
    v = np.zeros(p)
    eta = eta0 * np.ones(p)
    #
    # here: do one pass over dataset with inital x0?
    #
    start, end, score, passes = 0, 0, 0, 0
    scores = []
    while True:
        # prepare batches
        start = end
        end = start + btsz
        for item in batch_args:
            args[item] = batch_args[item][start:end]
        # Nic Schraudolph's complex number trick
        # see http://www.cs.toronto.edu/~vnair/ciar/mark_schmidt.pdf, p. 26
        # Why? This trick ensures that complex numbers _numerically_
        # behave like dual numbers (dual numbers: d**2 == 0, as opposed
        # to complex numbers, where i**2 = -1)
        sc, g = fandprime(x0 + ii*v, **args)
        #
        eta = eta * np.maximum(0.5, 1 + mu * v * np.real(g))
        # gradient step
        x0 -= eta * np.real(g)
        #
        v *= lmbd
        v += eta*(np.real(g) - lmbd*np.imag(g)/sys.float_info.min)
        score += sc
        # do some logging of error, eta, v, x0, g??
        if (end >= nos):
            # do a full pass over training data to determine 
            # training error with current parameters?
            #
            # start at beginning of data again
            end = 0
            if verbose:
                print "[smd] Epoch %d, Score %f" % (passes, score)
                #print np.min(eta), np.max(eta)
            scores.append(score)
            score = 0
            passes += 1
            if passes >= epochs:
                break
    return x0, scores


def msgd(x0, fandprime, args, batch_args,
        epochs, nos, lr, btsz, beta = 0.,
        verbose=False, **params):
    """
    Minibatch stochastic gradient descent.
    """
    start, end, score, passes = 0, 0, 0, 0
    scores = []
    # old direction for momentum
    _d = 0
    while True:
        # prepare batches
        start = end
        end = start + btsz
        for item in batch_args:
            args[item] = batch_args[item][start:end]
        # gradient + old direction is new direction
        sc, d = fandprime(x0, **args)
        d = -lr*d + beta * _d
        # descent
        x0 += d
        _d = d
        score += sc
        if (end >= nos):
            # start at beginning of data again
            end = 0
            if verbose:
                print "[msgd] Epoch %d, Score %f, lr %f, beta %f" % (passes, score, lr, beta)
            scores.append(score)
            score = 0
            passes += 1
            if passes >= epochs:
                break
    return x0, scores

def rmsprop(x0, fandprime, args, batch_args,
        epochs, nos, lr, btsz, decay = 0.9,
        verbose=False, **params):
    """
    Root mean squared stochastic gradient descent.
    (also see: Adagrad).
    """
    start, end, score, passes = 0, 0, 0, 0
    scores = []
    # old direction for momentum
    # moving average squared gradient
    masqg = 1.
    while True:
        # prepare batches
        start = end
        end = start + btsz
        for item in batch_args:
            args[item] = batch_args[item][start:end]
        # sc == score; d == gradient (_d_irection) 
        sc, d = fandprime(x0, **args)
        masqg *= decay
        masqg += (1 - decay) * (d**2)
        d = -(lr*d)/np.sqrt(masqg + 1e-8)
        # descent
        x0 += d
        score += sc
        if (end >= nos):
            # start at beginning of data again
            end = 0
            if verbose:
                print "[rmsprop] Epoch %d, Score %f, lr %f, decay %f" % (passes, score, lr, decay)
            scores.append(score)
            score = 0
            passes += 1
            if passes >= epochs:
                break
    return x0, scores


def lbfgsb(func, x0, fprime=None, args=(), approx_grad=0, 
        bounds=None, m=10, factr=10000000.0, pgtol=1e-05, 
        epsilon=1e-08, iprint=-1, maxfun=15000, disp=2, **params):
    """
    Limited Memory BFGS (with boundary conditions).
    A wrapper for lbfgsb from scipy.
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
    Nonlinear conjugate gradient descent.
    A wrapper for tnc from scipy. 
    """
    return fmin_tnc(func=func, x0=x0, fprime=fprime, args=args, 
            approx_grad=approx_grad, bounds=bounds, epsilon=epsilon, 
            scale=scale, offset=offset, messages=messages, 
            maxCGit=maxCGit, maxfun=maxfun, eta=eta, stepmx=stepmx, 
            accuracy=accuracy, fmin=fmin, ftol=ftol, xtol=xtol, 
            pgtol=pgtol, rescale=rescale, disp=disp)
