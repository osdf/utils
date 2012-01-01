"""
ICA -- Independent Component Analysis

This is based on 
ICA with Reconstruction Cost for Efficient
Overcomplete Feature Learning, by Quoc V. Le et al.

If you are looking for 'standard' ICA
you need to go somewhere else, e.g.
scikits.learn.
"""


import numpy as np
import scipy as sp


from losses import ssd
from misc import Dtable


def score(weights, structure, inputs,
        predict=False, error=False):
    """
    """
    ind = structure["ind"]
    hid = structure["hid"]
    # independent components (ic): linear projections
    ic = np.dot(inputs, weights.reshape(ind, hid))
    if predict:
        return ic 

    n, _ = inputs.shape
    z = np.dot(ic, weights.reshape(ind, hid).T)
    # smooth l1 penalty cost
    l1 = structure["l1"]
    pl1 = structure["lmbd"] * np.sum(l1(ic))
    if error:
        sc, err = ssd(z, inputs, predict=False, error=True)
        sc += pl1
        # returns first derivative of rec. error!
        return sc, err 
    else:
        sc = ssd(z, inputs, predict=False, error=False) +\
                structure["lmbd"]*pl1
        return sc


def score_grad(weights, structure, inputs, **params):
    """
    """
    # get _compelete_ score and first deriv of rec. error
    sc, delta = score(weights, inputs=inputs, structure=structure, 
            predict=False, error=True, **params)
    ind = structure["ind"]
    hid = structure["hid"]
    l1 = structure["l1"]
    #
    ic = np.dot(inputs, weights.reshape(ind, hid))
    Dsc_Dic = np.dot(delta, weights.reshape(ind, hid))
    #
    g = np.zeros(weights.shape, dtype=weights.dtype)
    g += np.dot(ic.T, delta).T.flatten()
    g += np.dot(inputs.T, Dsc_Dic).flatten()
    g += structure["lmbd"] * np.dot(inputs.T, Dtable[l1](ic)).flatten()
    return sc, g


def grad(weights, structure, inputs, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, **params)
    return g


def score_grad_norm(weights, structure, inputs, eps=10**-5, **params):
    """
    """
    ind = structure["ind"]
    hid = structure["hid"]
    w = weights.reshape(ind, hid)
    _l2 = sp.sqrt(np.sum(w ** 2, axis=0) + eps)
    _w = w/_l2
    # gradient from ball projection + reshaped
    sc, _g = score_grad(_w.flatten(), structure, inputs, **params)
    _g = _g.reshape(ind, hid)
    g = _g/_l2 - _w * (np.sum(_g * w, axis=0))/(_l2 **2)
    return sc, g.flatten()


def grad_norm(weights, structure, inputs, eps=10**-5, **params):
    """
    """
    _, g = score_grad_norm(weights, structure, inputs, eps=10**-5, **params)
    return g


def check_the_grad(nos=5000, ind=30, outd=10,
        eps=10**-4, verbose=False):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from opt import check_grad
    from misc import logcosh, sqrtsqr
    # number of input samples (nos)
    # with dimension ind each
    ins = np.random.randn(nos, ind)
    #
    weights = 0.001*np.random.randn(ind, outd).flatten()
    structure = dict()
    structure["l1"] = sqrtsqr
    structure["lmbd"] = 1 
    structure["ind"] = ind
    structure["hid"] = outd
    #
    cg = dict()
    cg["inputs"] = ins
    cg["structure"] = structure
    #
    delta = check_grad(score, grad, weights, cg, eps=eps, verbose=verbose)
    assert delta < 10**-2, "[ica.py] check_the_grad FAILED. Delta is %f" % delta
    return True


def test_cifar(gray, outd, lambd,
        opt, epochs=10, btsz=100,
        lr=1e-8, beta=0.9,
        eta0=2e-6, mu=0.02, lmbd=0.99,
        w=None):
    """
    """
    from opt import msgd, smd
    from misc import logcosh, sqrtsqr
    #
    n, ind = gray.shape
    if w is None:
        if opt is smd:
            # needs complex initialization
            weights = np.zeros((ind*outd), dtype=np.complex)
            weights[:] = 0.001 * np.random.randn(ind, outd).flatten()
        else:
            weights = 0.001 * np.random.randn(ind, outd).flatten()
    else:
        print "Continue with provided weights w."
        weights = w
        #
    structure = dict()
    structure["l1"] = sqrtsqr
    # function parameter lmbd is for SMD!!
    structure["lmbd"] = lambd
    structure["ind"] = ind
    structure["hid"] = outd
    #
    print "Training starts ..."
    params = dict()
    params["x0"] = weights
    if opt is msgd or opt is smd:
        params["fandprime"] = score_grad_norm
        params["nos"] = gray.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": gray}
        params["epochs"] = epochs
        params["btsz"] = btsz
        # msgd
        params["lr"] = lr 
        params["beta"] = beta
        # smd
        params["eta0"] = eta0
        params["mu"] = mu
        params["lmbd"] = lmbd
        #
        params["verbose"] = True
    else:
        # opt from scipy
        params["func"] = score
        params["fprime"] = grad_norm
        params["args"] = (structure, gray)
        params["maxfun"] = epochs
        params["m"] = 50
        params["factr"] = 10.
    weights = opt(**params)[0]
    print "Training done."
    #
    return weights
