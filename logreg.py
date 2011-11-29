"""
Multinomial Logistic Regression.
"""


import numpy as np
import scipy.linalg as la


from misc import logsumexp
from losses import score_xe


def score(weights, inputs, targets=None, 
        predict=False, error=False, **params):
    """
    """
    _, di = inputs.shape
    dt = weights.shape[0]/(di + 1)
    z = np.dot(inputs, weights[:di*dt].reshape(di, dt)) + weights[di*dt:]
    return score_xe(z, targets=targets, predict=predict, error=error)


def predict(weights, inputs, **params):
    """
    """
    return score(weights, inputs, targets=None, predict=True)


def grad(weights, inputs, targets, **params):
    """
    Compute the (batch) gradient at _weights_
    for training set _inputs_/_targets_.
    _lmbd_ is weight decay factor.
    """
    n, di = inputs.shape
    dt = np.max(targets) + 1
    g = np.zeros(weights.shape)
    xe, error = score(weights, inputs, targets, predict=False, error=True)
    # one signal per input sample
    g[:di*dt] = np.dot(inputs.T, error).flatten()
    g[di*dt:] = error.sum(axis=0)
    return g


def check_the_grad(nos=1000, ind=30, classes=5, eps=10**-6):
    """
    """
    from opt import check_grad
    #
    weights = 0.1*np.random.randn(ind*classes + classes)
    ins = np.random.randn(nos, ind)
    outs = np.random.random_integers(classes, size=(nos)) - 1
    #
    cg = dict()
    cg["inputs"] = ins
    cg["targets"] = outs
    #
    delta = check_grad(score, grad, weights, cg, eps)
    assert delta < 10**-4, "[logreg.py] check_the_gradient FAILED. Delta is %f" % delta
    return True

def demo_mnist(epochs, lr, btsz, opt):
    """
    """
    from misc import sigmoid, load_mnist
    from losses import score_xe, loss_zero_one
    from opt import msgd
    #
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    test_in, test_tar = testset
    #
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    # setup weights
    weights = 0.01 * np.random.randn(di*dt+dt)
    weights[-dt:] = 0.
    print "Training starts..."
    params = dict()
    params["func"] = score
    params["x0"] = weights
    params["fprime"] = grad
    params["inputs"] = inputs
    params["targets"] = targets
    params["epochs"] = epochs
    params["lr"] = lr 
    params["btsz"] = btsz
    params["verbose"] = True
    if opt is msgd:
        params["nos"] = inputs.shape[0]
        params["args"] = {}
        params["batch_args"] = {"inputs": inputs, "targets": targets}
    else:
        params["args"] = (inputs, targets)
        params["maxfun"] = epochs 
    weights = opt(**params)[0]
    print loss_zero_one(predict(weights, test_in), test_tar)
    return
