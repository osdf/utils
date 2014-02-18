"""
Standard neural network.
"""


import numpy as np
import scipy.linalg as la


from misc import Dtable


def score(weights, structure, inputs, targets,
        predict=False, error=False, **params):
    """
    Compute the score of a neural network specified in _structure_.
    """
    layers = structure["layers"]
    activs = structure["activs"]
    z = inputs
    if error:
        hiddens = [z]
    idx = 0
    for l, A in zip(layers, activs):
        idy = idx + l[0]*l[1]
        z = A(np.dot(z, weights[idx:idy].reshape(l[0], l[1])) + weights[idy:idy+l[1]])
        idx = idy+l[1]
        if error:
            # note: store a pointer to z, i.e.
            # values _after_ applying activation A!
            hiddens.append(z)
    
    if error:
        structure["hiddens"] = hiddens
    
    wdecay = structure["l2"] * np.sum(weights**2)
    sc = structure["score"]
    return sc(z, targets, predict=predict, error=error, addon=wdecay)


def score_grad(weights, structure, inputs, targets, **params):
    """
    Compute the score and the gradient for a set of inputs/targets.
    """
    g = np.zeros(weights.shape, dtype=weights.dtype)
    # forward pass through model,
    # need 'error' signal at the end.
    sc, delta = score(weights, inputs=inputs, targets=targets,
            structure=structure, predict=False, error=True, **params)
    layers = structure["layers"]
    activs = structure["activs"]
    hiddens = structure["hiddens"]

    tmp = hiddens[-1]
    idy = 0
    idx = 0

    # Filling up gradient from _top_ layer downwards!
    for l, A, h in reversed(zip(layers, activs, hiddens[:-1])):
        # tmp are values _after_ applying 
        # activation A to matrix-vector product
        # Compute dE/da (small a -> before A is applied)
        dE_da = delta * (Dtable[A](tmp))
        idx = idy + l[1]
        # gradient for biases
        if idy == 0:
            g[-idx:] = dE_da.sum(axis=0)
        else:
            g[-idx:-idy] = dE_da.sum(axis=0)

        idy = idx + l[0] * l[1]
        # gradient for weights in layer l
        g[-idy:-idx] = np.dot(h.T, dE_da).ravel()
        # backprop delta
        delta = np.dot(delta, weights[-idy:-idx].reshape(l[0], l[1]).T)
        tmp = h

    g += 2*structure["l2"]*weights
    # clean up structure
    del structure["hiddens"]
    n, _ = inputs.shape
    return sc, g


def grad(weights, structure, inputs, targets, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, targets, **params)
    return g


def predict(weights, structure, inputs, **params):
    """
    """
    return score(weights, structure, inputs, 
            targets=None, predict=True)


def init_weights(structure, var=0.01):
    """
    """
    size=0
    layers = structure["layers"]
    # determine number of total weights
    for l in layers:
        size += l[0]*l[1] + l[1]
    weights = np.zeros(size)
    # init weights. biases are 0. 
    idx = 0
    for l in layers:
        weights[idx:idx+l[0]*l[1]] = var * np.random.randn(l[0]*l[1])
        idx += l[0] * l[1] + l[1]
    return weights


def check_the_grad(regression=True, nos=1, ind=30,
        outd=3, bxe=False, eps=1e-8, verbose=False):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from misc import sigmoid, identy, check_grad
    from losses import xe, ssd, mia
    # number of input samples (nos)
    # with dimension ind each
    ins = np.random.randn(nos, ind)
    #
    structure = dict()
    if regression:
    # Regression
    # Network with one hidden layer
        structure["layers"] = [(ind, 15), (15, outd)]
        # the last layer also needs to be specified.
        # it should be a linear layer (see score functions in losses.py).
        structure["activs"] = [np.tanh, identy]
        structure["score"] = ssd 
        outs = np.random.randn(nos, outd)
    else:
        # Classification
        # _outd_ is interpreted as number of classes
        structure["layers"] = [(ind, 15), (15, outd)]
        # the last layer also needs to be specified.
        structure["activs"] = [sigmoid, identy]
        if bxe:
            structure["score"] = mia
            outs = 1.*(np.random.rand(nos, outd) > 0.5)
        else:
            structure["score"] = xe
            outs = np.random.random_integers(outd, size=(nos)) - 1
    # weight decay
    structure["l2"] = 0.1
    weights = init_weights(structure)
    cg = dict()
    cg["inputs"] = ins
    cg["targets"] = outs
    cg["structure"] = structure
    #
    delta = check_grad(score, grad, weights, cg, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[nn.py] check_the_grad FAILED. Delta is %f" % delta
    return True


def demo_mnist(hiddens, opt, l2=1e-6, epochs=10, 
        lr=1e-4, beta=0., decay=0.9,btsz=128, eta0 = 0.0005, 
        mu=0.02, lmbd=0.99, weightvar=0.01, 
        w=None):
    """
    """
    from misc import sigmoid, identy, load_mnist
    from losses import xe, zero_one
    from opt import msgd, smd, rmsprop
    #
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    test_in, test_tar = testset
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    structure = {}
    structure["layers"] = [(di, hiddens), (hiddens, dt)]
    structure["activs"] = [np.tanh, identy]
    structure["score"] = xe
    structure["l2"] = l2
    # get weight initialized
    if w is None:
        weights = init_weights(structure, weightvar)
        if opt is smd:
            # needs complex weights
            weights = np.asarray(weights, dtype=np.complex)
    else:
        print "Continue with provided weights w."
        weights = w
    #
    print "Training starts..."
    params = dict()
    params["x0"] = weights
    params["fandprime"] = score_grad
    if opt is msgd or opt is smd or opt is rmsprop:
        params["nos"] = inputs.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": inputs, "targets": targets}
        params["epochs"] = epochs
        params["btsz"] = btsz
        params["verbose"] = True
        # for msgd
        params["lr"] = lr
        params["beta"] = beta
        params["decay"] = decay
        # for smd
        params["eta0"] = eta0
        params["mu"] = mu
        params["lmbd"] = lmbd
    else:
        params["args"] = (structure, inputs, targets)
        params["maxfun"] = epochs
        # for lbfgs
        params["m"] = 25
    
    weights = opt(**params)[0]
    print "Training done."
    
    # Evaluate on test set
    test_perf = zero_one(predict(weights, structure, test_in), test_tar)
    print "Test set performance:", test_perf
    return weights
