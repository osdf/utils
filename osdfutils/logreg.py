"""
Multinomial Logistic Regression.
"""


import numpy as np
import scipy.linalg as la


from misc import logsumexp
from losses import xe, mia


def score_mia(weights, inputs, targets=None,
        predict=False, error=False, **params):
    """
    Score for multiple independent (output) attributes.

    This allows 'standard' logistic regression (one output,
    two classes).
    """
    _, di = inputs.shape
    dt = weights.shape[0]/(di + 1)
    z = np.dot(inputs, weights[:di*dt].reshape(di, dt)) + weights[di*dt:]
    return mia(z, targets=targets, predict=predict, error=error)


def score_xe(weights, inputs, targets=None, 
        predict=False, error=False, **params):
    """
    """
    _, di = inputs.shape
    dt = weights.shape[0]/(di + 1)
    z = np.dot(inputs, weights[:di*dt].reshape(di, dt)) + weights[di*dt:]
    return xe(z, targets=targets, predict=predict, error=error)


def predict(weights, inputs, **params):
    """
    """
    return score(weights, inputs, targets=None, predict=True)


def score_grad_xe(weights, inputs, targets, **params):
    """
    Compute the (batch) gradient at _weights_
    for training set _inputs_/_targets_.
    _lmbd_ is weight decay factor.
    """
    n, di = inputs.shape
    dt = weights.shape[0]/(di + 1)
    g = np.zeros(weights.shape, dtype=weights.dtype)
    xe, error = score_xe(weights, inputs, targets, predict=False, error=True)
    # one signal per input sample
    g[:di*dt] = np.dot(inputs.T, error).ravel()
    g[di*dt:] = error.sum(axis=0)
    return xe, g


def grad_xe(weights, inputs, targets, **params):
    """
    """
    _, g = score_grad_xe(weights, inputs, targets, **params)
    return g


def score_grad_mia(weights, inputs, targets, **params):
    """
    Compute the (batch) gradient at _weights_
    for training set _inputs_/_targets_.
    _lmbd_ is weight decay factor.
    """
    _, di = inputs.shape
    dt = weights.shape[0]/(di + 1)
    g = np.zeros(weights.shape, dtype=weights.dtype)
    xe, error = score_mia(weights, inputs, targets, predict=False, error=True)
    # one signal per input sample
    g[:di*dt] = np.dot(inputs.T, error).ravel()
    g[di*dt:] = error.sum(axis=0)
    return xe, g


def grad_mia(weights, inputs, targets, **params):
    """
    """
    _, g = score_grad_mia(weights, inputs, targets, **params)
    return g


def check_the_grad(nos=1, ind=30, outd=5, bxe=False,
        eps=1e-6, verbose=False):
    """
    """
    from opt import check_grad
    #
    weights = 0.1 * np.random.randn(ind*outd + outd)
    ins = np.random.randn(nos, ind)

    if bxe:
        score = score_mia
        grad = grad_mia
        outs = 1.*(np.random.rand(nos, outd) > 0.5)
    else:
        outs = np.random.random_integers(outd, size=(nos)) - 1
        score = score_xe
        grad = grad_xe

    cg = dict()
    cg["inputs"] = ins
    cg["targets"] = outs
    #
    delta = check_grad(f=score, fprime=grad, x0=weights,
            args=cg, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[logreg.py] check_the_gradient FAILED. Delta is %f" % delta
    return True


def demo_mnist(opt, epochs=10, btsz=100,
        lr = 0.1, beta = 0.9,
        eta0 = 0.0005, mu=0.02, lmbd=0.99,
        w=None):
    """
    """
    from misc import load_mnist
    from losses import zero_one
    from opt import msgd, smd
    #
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    test_in, test_tar = testset
    #
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    # setup weights
    if w is None:
        if opt is smd:
            # needs complex initialization
            weights = np.zeros((di*dt+dt), dtype=np.complex)
            weights[:] = 0.001 * np.random.randn(di*dt+dt)
        else:
            weights = np.zeros((di*dt+dt), dtype=np.complex)
            weights = 0.* np.random.randn(di*dt+dt)
        weights[-dt:] = 0.
    else:
        print "Continue with provided weights w."
        weights = w
    #
    print "Training starts..."
    params = dict()
    params["x0"] = weights
    if opt is msgd or opt is smd:
        params["fandprime"] = score_grad
        params["nos"] = inputs.shape[0]
        params["args"] = {}
        params["batch_args"] = {"inputs": inputs, "targets": targets}
        params["epochs"] = epochs
        params["btsz"] = btsz
        # msgd
        params["beta"] = beta
        params["lr"] = lr 
        # smd
        params["eta0"] = eta0
        params["mu"] = mu
        params["lmbd"] = lmbd
        params["verbose"] = True
    else:
        # opt from scipy
        params["func"] = score
        params["fprime"] = grad
        params["args"] = (inputs, targets)
        params["maxfun"] = epochs
        params["m"] = 50
    weights = opt(**params)[0]
    print "Training done."
    #
    print "Test set preformance:",\
            zero_one(predict(weights, test_in), test_tar)
    return weights
