"""
Multinomial Logistic Regression.
"""


import numpy as np
import scipy.linalg as la


def logsumexp(array, axis):
    """
    Compute log of (sum of exps) 
    along _axis_ in _array_ in a 
    stable way.
    """
    axis_max = np.max(array, axis)[:, np.newaxis]
    return axis_max + np.log(np.sum(np.exp(array-axis_max), axis))[:, np.newaxis]


def score(weights, inputs, targets, lmbd):
    """
    Compute score for _weights_, given
    _inputs_ and _targets_, the provided 
    supervision, need to be in 1-of-K coding.
    """
    n, di = inputs.shape
    dt = np.max(targets) + 1
    # log predict is ok here -> loglikelihood
    pred = predict_log(weights, inputs, dt)
    # weight decay term
    reg = lmbd*np.sum(weights[:di*dt]**2)
    # negative loglikelihood + regularizer.
    # advanced indexing: only log predict at 
    # target class is counted (assumes: targets
    # as only one true target per input)
    print  -np.sum(pred[xrange(n), targets]) + reg
    return -np.sum(pred[xrange(n), targets]) + reg


def predict_log(weights, inputs, dt):
    """
    Predict target distribution 
    for _inputs_, given _weights_.
    """
    _, di = inputs.shape
    y = np.dot(inputs, weights[:di*dt].reshape(di, dt)) + weights[di*dt:]
    return y - logsumexp(y, axis=1)


def classify(weights, inputs, dt):
    """
    Classify _inputs_ given and model _weights_.

    Returns numbered classes
    """
    # np.exp() is not necessary,
    # because it is strictly increasing.
    y = predict_log(weights, inputs, dt)
    return np.argmax(y, axis=1) 


def grad(weights, inputs, targets, lmbd):
    """
    Compute the (batch) gradient at _weights_
    for training set _inputs_/_targets_.
    _lmbd_ is weight decay factor.
    """
    n, di = inputs.shape
    dt = np.max(targets) + 1
    delta = np.zeros(weights.shape)
    # The true predicted probabilities
    # are necessary here, thus np.exp(...)
    pred = np.exp(predict_log(weights, inputs, dt))
    error = pred
    # one signal per input sample
    pred[xrange(n), targets] -= 1
    delta[:di*dt] = np.dot(inputs.T, error).flatten() + 2*lmbd*weights[:di*dt]
    delta[di*dt:] = error.sum(axis=0)
    return delta


def sgd(weights, inputs, targets, epochs, lr, btsz, lmbd):
    """
    """
    n, _ = inputs.shape
    div = n/btsz
    mod = n%btsz
    lr /= btsz
    scores = []
    for e in xrange(epochs):
        sc = 0
        for b in xrange(div):
            sc += score(weights, inputs[b*btsz:(b+1)*btsz],
                    targets[b*btsz:(b+1)*btsz], lmbd)
            delta = grad(weights, inputs[b*btsz:(b+1)*btsz], 
                    targets[b*btsz:(b+1)*btsz], lmbd)
            weights -= lr*delta
        if mod>0:
            sc += score(weights, inputs[-mod:],targets[-mod:], lmbd)
            delta = grad(weights, inputs[-mod:], targets[-mod:], lmbd)
            weights -= lr*btsz*delta/mod
        scores.append(sc)
    lr *= btsz
    return scores


def one2K(classes):
    """
    Given a numeric encoding of
    class membership in _classes_, 
    produce a 1-of-K encoding.
    """
    c = classes.max() + 1
    targets = np.zeros((len(classes), c))
    targets[classes] = 1
    return targets


def K2one(targets):
    return np.argmax(targets)


def testing(nos, di, classes, epochs, lr, btsz, lmbd):
    """
    """
    samples = 2*np.random.randn(nos, di)
    targets = np.random.random_integers(classes, size=(nos))
    targets += np.random.random_integers(classes/2, size=(nos))
    targets -= 1
    c = np.max(targets)+1
    #targets = one2K(c)
    # init weights, bias have zeros
    weights = 0.01 * np.random.randn(di*c + c)
    weights[-classes:] = 0.
    sc = sgd(weights, samples, targets, epochs, lr, btsz, lmbd)
    return sc


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


def testing_mnist(epochs=80, lr=0.13, btsz=600, lmbd=0.0001):
    """
    """
    import scipy.optimize as opt
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    # setup weights
    weights = 0.01 * np.random.randn(di*dt+dt)
    weights[-dt:] = 0.
    print "Training starts..."
    #sc = sgd(weights, inputs, targets, epochs, lr, btsz, lmbd)
    #sc = opt.fmin_tnc(score, weights, grad, args=(inputs, targets, 0.0001), maxCGit=0)
    #sc = opt.fmin_l_bfgs_b(score, weights, grad, args=(inputs, targets, 0.0001), disp=2)
    #sc = opt.fmin_cg(score, weights, grad, args=(inputs, targets, 0.0001), retall=True)
    return sc

 
def load_mnist():
    """
    Get standard MNIST: training, validation, testing.
    File 'mnist.pkl.gz' must be in path.
    Download it from ...
    """
    import gzip, cPickle
    print "Reading mnist.pkl.gz ..."
    f = gzip.open('mnist.pkl.gz','rb')
    trainset, valset, testset = cPickle.load(f)
    return trainset, valset, testset


def loss(weights, inputs, targets):
    """
    """


