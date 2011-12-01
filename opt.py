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
    lr /= btsz
    #
    start = 0
    end = 0
    score = 0
    scores = []
    passes = 0
    while True:
        # prepare batches
        start = end
        end = start + btsz
        for item in batch_args:
            args[item] = batch_args[item][start:end]
        # do the work
        score += func(x0, **args)
        delta = fprime(x0, **args)
        x0 -= lr*delta
        scores.append(score)
        if (end >= nos):
            # start at beginning of data again
            end = 0
            if verbose:
                print "Epoch %d, Score %f" % (passes, score)
            scores.append(score)
            score = 0
            passes += 1
            if passes >= epochs:
                break
    lr *= btsz
    return x0, scores


def olbfgs(func, x0, fprime, args, batch_args,
        eta_0, m, tau, epochs, nos, btsz, verbose=False, 
        **params):
    """
    """
    SMALL = 10**-8
    S = np.zeros((m, x0.shape[0]))
    Y = np.zeros((m, x0.shape[0]))
    rho = np.zeros(m)
    alpha = np.zeros(m)
    index = -1
    s = 0
    y = 0
    iters = 0
    start = 0
    end = 0
    score = 0
    scores = []
    passes = 0
    while True:
        # get batch borders
        start = end
        end = start+btsz
        # get batch
        for item in batch_args:
            args[item] = batch_args[item][start:end]
        score += func(x0, **args)
        # compute update direction
        grad = fprime(x0, **args)
        if iters > 0:
            eta = eta_0 * tau/(tau + iters)
            #print 'eta', eta
            p = eta * grad
            S[index] = s
            Y[index] = y
            sy = np.dot(s, y)
            yy = np.dot(y, y)
            #print 'sx, yy', sy, yy
            rho[index] = 1./sy
            #
            cap = min(m, iters)
            #
            alpha *= 0
            #
            counter = 0
            i = index
            while counter < cap:
                counter += 1
                alpha[i] = rho[i] * np.dot(S[i], p)
                #print 'rho, alpha', rho[i], alpha[i]
                p -= alpha[i] * Y[i]
                i = (i-1)%m
            #print
            #
            p *= sy/yy 
            #
            counter = 0
            i = (index - (cap-1)) % m
            while counter < cap:
                counter += 1
                beta = rho[i] * np.dot(Y[i], p)
                p += (alpha[i] - beta) * S[i]
                i = (i+1)%m
            s = p
        else:
            s = -SMALL * grad
        #
        x0 += s
        #
        y = fprime(x0, **args)
        y -= grad
        #
        iters += 1
        index = (index + 1)%m
        del grad
        if (end >= nos):
            # start at beginning of data again
            end = 0
            if verbose:
                print passes, score
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
