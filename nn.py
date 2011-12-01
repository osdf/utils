"""
Standard neural networks
"""


import numpy as np
import scipy.linalg as la


from misc import logsumexp, Dtable 


def score(weights, structure, inputs, targets,
        predict=False, error=False, **params):
    """
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
            # values _after_ applying activation A!
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
    sc = structure["score"]
    return sc(z, targets, predict=predict, error=error)


def grad(weights, structure, inputs, targets, **params):
    """
    """
    g = np.zeros(weights.shape)
    # forward pass through model,
    # need 'error' signal at the end.
    sc, delta = score(weights, inputs=inputs, targets=targets,
            structure=structure, predict=False, error=True, **params)
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
        # Compute dE/da (small a -> before A is applied)
        dE_da = delta * (Dtable[A](tmp))
        idx = idy + l[1]
        # gradient for biases
        g[-idx:-idy] = dE_da.sum(axis=0)
        idy = idx + l[0] * l[1]
        # gradient for weights in layer l
        g[-idy:-idx] = np.dot(h.T, dE_da).flatten()
        # backprop delta
        delta = np.dot(delta, weights[-idy:-idx].reshape(l[0], l[1]).T)
        tmp = h
    # clean up structure
    del structure["hiddens"]
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


def demo_mnist(hiddens, epochs, lr, btsz, lmbd, opt):
    """
    """
    from misc import sigmoid, load_mnist
    from losses import score_xe, loss_zero_one
    from opt import msgd, olbfgs
    #
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    test_in, test_tar = testset
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    structure = {}
    structure["layers"] = [(di, hiddens), (hiddens, dt)]
    structure["activs"] = [np.tanh]
    structure["score"] = score_xe
    weights = init_weights(structure) 
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
    if opt is olbfgs:
        params["nos"] = inputs.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": inputs, "targets": targets}
        params["eta_0"] = lr 
        params["tau"] = 100.
        params["m"] = 10 
    else:
        params["args"] = (structure, inputs, targets)
        params["maxfun"] = epochs 
    weights = opt(**params)[0]
    print loss_zero_one(predict(weights, structure, test_in), test_tar)
    return


def check_the_grad(regression=True, nos=5000, ind=30, 
        outd=3, classes=10, eps=10**-4, verbose=False):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from opt import check_grad
    from misc import sigmoid
    from losses import score_xe, score_ssd
    # number of input samples (nos)
    # with dimension d each
    ins = np.random.randn(nos, ind)
    #
    structure = dict()
    if regression:
    # Regression    
    # Network with one hidden layer
        structure["layers"] = [(ind, 15), (15, outd)]
        structure["activs"] = [np.tanh]
        structure["score"] = score_ssd
        outs = np.random.randn(nos, outd)
    else:
        # Classification
        classes = 10
        structure["layers"] = [(ind, 15), (15, classes)]
        structure["activs"] = [sigmoid]
        structure["score"] = score_xe
        outs = np.random.random_integers(classes, size=(nos)) - 1
    weights = init_weights(structure) 
    cg = dict()
    cg["inputs"] = ins
    cg["targets"] = outs
    cg["structure"] = structure
    #
    delta = check_grad(score, grad, weights, cg, eps=eps, verbose=verbose)
    assert delta < 10**-2, "[nn.py] check_the_grad FAILED. Delta is %f" % delta
    return True
