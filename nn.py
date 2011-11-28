"""
Standard neural networks
"""


import numpy as np
import scipy.linalg as la


from misc import logsumexp, Dtable, sigmoid


def score_xe(z, targets, predict=False, error=False):
    """
    """
    if predict:
        return np.argmax(z, axis=1)
    #
    _xe = z - logsumexp(z, axis=1)
    n, _ = _xe.shape
    xe = -np.sum(_xe[np.arange(n), targets])
    if error:
        err = np.exp(_xe)
        err[np.arange(n), targets] -= 1
        #score + error
        return xe, err
    else:
        return xe


def score_ssd(z, targets, predict=False, error=False):
    """
    """
    if predict:
        return z
    #
    err = z - targets
    if error:
        # score + error
        return 0.5*np.sum(err**2), err
    else:
        # only return score
        return 0.5*np.sum(err**2)


def score_mia():
    """
    Multiple independent attributes.
    """
    pass


def fward(weights, structure, inputs, targets, predict=False, error=False):
    """
    structure: dictrionary with
    layers - list of pairs
    activations functions - list of act. func., apart from last layer - id
    score - list? 
    """
    layers = structure["layers"]
    activs = structure["activs"]
    if error:
        hiddens = []
    z = inputs
    idx = 0
    for l, A in zip(layers[:-1], activs):
        if error:
            # note: store a pointer to z, i.e.
            # values _after_ applying activation a!
            hiddens.append(z)
        idy = idx + l[0]*l[1]
        z = A(np.dot(z, weights[idx:idy].reshape(l[0], l[1])) \
                + weights[idy:idy+l[1]])
        idx = idy+l[1]
    if error:
        hiddens.append(z)
        structure["hiddens"] = hiddens
    # last layer: special care
    l = layers[-1]
    idy = idx + l[0]*l[1]
    # no activation function, 
    # everyting will be handeled by score
    z = np.dot(z, weights[idx:idy].reshape(l[0], l[1])) \
                + weights[idy:idy+l[1]]
    score = structure["score"]
    return score(z, targets, predict=predict, error=error)


def grad(weights, structure, inputs, targets):
    """
    """
    g = np.zeros(weights.shape)
    # forward pass through model,
    # need 'error' signal at the end.
    score, delta = fward(weights, structure, inputs, targets, 
            predict=False, error=True)
    layers = structure["layers"]
    activs = structure["activs"]
    hiddens = structure["hiddens"]
    # backprop through last layer, activation fct. is id.
    l = layers[-1]
    idx = l[1]
    g[-idx:] = delta.sum(axis=0)
    idy = idx + l[0]*l[1]
    g[-idy:-idx] = np.dot(hiddens[-1].T, delta).flatten()
    # new delta for one layer below
    delta = np.dot(delta, weights[-idy:-idx].reshape(l[0], l[1]).T)
    tmp = hiddens[-1]
    for l, A, h in reversed(zip(layers[:-1], activs, hiddens[:-1])):
        # remember: tmp are values _after_ applying 
        # activation A to matrix-vector product
        # Compute dE/dA
        dE_dA = delta * (Dtable[A](tmp))
        idx = idy + l[1]
        # gradient for biases
        g[-idx:-idy] = dE_dA.sum(axis=0)
        idy = idx + l[0] * l[1]
        # gradient for weights in layer l
        g[-idy:-idx] = np.dot(h.T, dE_dA).flatten()
        # backprop delta
        delta = np.dot(delta, weights[-idy:-idx].reshape(l[0], l[1]).T)
        tmp = h
    # clean up structure
    del structure["hiddens"]
    return g


def predict(weights, structure, inputs):
    """
    """
    return fward(weights, structure, inputs, 
            targets=None, predict=True)


def demo(hiddens, epochs, lr, btsz, lmbd):
    """
    """
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    structure = {}
    structure["layers"] = [(di, hiddens), (hiddens, dt)]
    structure["activs"] = [sigmoid]
    strucutre["score"] = score_sm


def check_the_gradient(nos=5000, ind=30, outd=3, classes=10):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from opt import check_grad
    # number of input samples (nos)
    # with dimension d each
    ins = np.random.randn(nos, ind)
    #
    # Uncomment for
    # Regression    
    # Network with one hidden layer
    #structure = dict()
    #structure["layers"] = [(ind, 15), (15, outd)]
    #structure["activs"] = [sigmoid]
    #structure["score"] = score_ssd
    #outs = np.random.randn(nos, outd)
    #weights = np.random.randn(ind*15 + 15 + 15*outd + outd)
    # Uncomment for
    # Classification
    classes = 10
    structure = dict()
    structure["layers"] = [(ind, 15), (15, classes)]
    structure["activs"] = [sigmoid]
    structure["score"] = score_xe
    outs = np.random.random_integers(classes, size=(nos)) - 1
    weights = np.random.randn(ind*15 + 15 + 15*classes + classes)
    #
    # Same for both Regression/Classification
    cg = dict()
    cg["inputs"] = ins
    cg["targets"] = outs
    cg["structure"] = structure
    #
    delta = check_grad(fward, grad, weights, cg)
    print delta
