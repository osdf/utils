"""
Linearly augmented feed forward network,
Solving the Ill-Conditioning in Neural Network Learning,
by Patrick van der Smagt and Gerd Hirzinger,
http://www.brml.de/uploads/tx_sibibtex/SmaHir1998a.pdf
"""


import numpy as np


from misc import Dtable


def score(weights, structure, inputs, targets, predict=False, 
        error=False, **params):
    """
    """
    hdim = structure["hdim"]
    odim = structure["odim"]
    A = structure["af"]
    _, idim = inputs.shape
    ih = idim * hdim
    ho = hdim * odim
    flann = np.dot(inputs, weights[:ih].reshape(idim, hdim))
    hddn = A(flann + weights[ih:ih+hdim])
    wdecay = np.sum(weights[:ih]**2)

    if error:
        structure["hiddens"] = hddn
    z = np.dot(hddn, weights[ih+hdim:ih+hdim+ho].reshape(hdim, odim)) + weights[-odim:]
    sc = structure["score"]
    flann = np.atleast_2d(flann.sum(axis=1)).T
    wdecay += np.sum(weights[ih+hdim:ih+hdim+ho]**2)
    return sc(z+flann, targets, predict=predict, error=error, addon=structure["l2"]*wdecay)


def score_grad(weights, structure, inputs, targets, **params):
    """
    """
    hdim = structure["hdim"]
    odim = structure['odim']
    _, idim = inputs.shape
    ih = idim * hdim
    ho = hdim * odim
    af = structure["af"]
    g = np.zeros(weights.shape, dtype=weights.dtype)
    # forward pass through model,
    # need 'error' signal at the end.
    sc, delta = score(weights, structure=structure, inputs=inputs, targets=targets, predict=False, error=True, **params)
    # recover saved hidden values
    hddn = structure["hiddens"]
    g[ih+hdim:ih+hdim+ho] = np.dot(hddn.T, delta).ravel()
    g[ih+hdim:ih+hdim+ho] += 2*structure["l2"]*weights[ih+hdim:ih+hdim+ho]
    g[-odim:] = delta.sum(axis=0)
    dsc_dha = np.dot(delta, weights[ih+hdim:ih+hdim+ho].reshape(hdim, odim).T) * Dtable[af](hddn)
    g[:ih] += np.dot(inputs.T, dsc_dha).ravel()
    # flann
    dsum = np.atleast_2d(delta.sum(axis=1)).T
    dflann = (inputs*dsum).sum(axis=0)
    g[:ih] += np.repeat(dflann, hdim)
    g[:ih] += 2*structure["l2"]*weights[:ih]
    g[ih:ih+hdim] = dsc_dha.sum(axis=0)
    # clean up structure
    del structure["hiddens"]
    return sc, g


def grad(weights, structure, inputs, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, **params)
    return g


def predict(weights, structure, inputs, **params):
    """
    """
    return score(weights, structure, inputs, targets=None, predict=True)


def check_the_grad(loss, nos=1, idim=30, hdim=10, odim=5, eps=1e-8, verbose=False):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from opt import check_grad
    from misc import sigmoid
    from losses import ssd, mia, xe
    # number of input samples (nos)
    # with dimension ind each
    ins = np.random.rand(nos, idim)
    if loss is ssd:
        outs = np.random.randn(nos, odim)
    if loss is mia:
        outs = 1.*(np.random.rand(nos, odim) > 0.5)
    if loss is xe:
        outs = np.random.random_integers(odim, size=(nos))-1
    structure = dict()
    structure["hdim"] = hdim
    structure["odim"] = odim
    structure["af"] = sigmoid
    structure["score"] = loss
    structure["l2"] = 1e-2

    weights = np.zeros(idim*hdim + hdim + hdim*odim + odim)
    weights[:idim*hdim] = 0.001 * np.random.randn(idim*hdim)
    weights[idim*hdim+hdim:-odim] = 0.001 * np.random.randn(hdim*odim)
    
    args = dict()
    args["inputs"] = ins
    args["targets"] = outs
    args["structure"] = structure
    #
    delta = check_grad(score, grad, weights, args, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[flann.py] check_the_grad FAILED. Delta is %f" % delta
    return True


def demo_mnist(hiddens, opt, l2=1e-6, epochs=10, 
        lr=1e-4, beta=0., btsz=128, eta0 = 0.0005, 
        mu=0.02, lmbd=0.99, weightvar=0.01, 
        w=None):
    """
    """
    from misc import sigmoid, load_mnist
    from losses import xe, zero_one
    from opt import msgd, smd 
    #
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    test_in, test_tar = testset
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    structure = {}
    structure["hdim"] = hiddens
    structure["odim"] = dt
    structure["af"] = np.tanh
    structure["score"] = xe
    structure["l2"] = l2
    # get weight initialized
    if w is None:
        weights = np.zeros(di*hiddens + hiddens + hiddens*dt + dt)
        weights[:hiddens*di] = 0.001 * np.random.randn(di*hiddens)
        weights[hiddens*(di+1):-dt] = 0.001 * np.random.randn(hiddens*dt)
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
    if opt is msgd or opt is smd:
        params["nos"] = inputs.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": inputs, "targets": targets}
        params["epochs"] = epochs
        params["btsz"] = btsz
        params["verbose"] = True
        # for msgd
        params["lr"] = lr
        params["beta"] = beta
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
