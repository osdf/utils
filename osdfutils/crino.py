"""
Some theano offsprings.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


def skmeans():
    """
    synchronous k-means.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX),
        borrow=True, name='W')
    sprod = T.dot(x, W)

    cost = T.sum((X - np.dot(sprod, W.T))**2)
    grads = T.grad(cost, W)


def sae():
    """
    synchronous autoencoder.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX),
        borrow=True, name='W')
    h = T.dot(x, W)
    sh = h*h
    rec = T.sum(x - T.dot(sh*h, W.T), axis=1)
    cost = T.mean(rec)
    grads = T.grad(cost, W)


def zbae(Winit, activ='TRec', theta=1.):
    """
    Zero bias autoencoder.
    See Zero-bias autoencoders and the benefits of co-adapting features,
    by Memisevic, R., Konda, K., Krueger, D.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX),
        borrow=True, name='W')
    _b = np.zeros((Winit.shape[0],), dtype=theano.config.floatX)
    b = theano.shared(value=_b, borrow=True)

    h = T.dot(x, W)
    if activ is "TRec":
        print "Using TRec as activation"
        h = h * (h > theta)
    else:
        print "Using TLin as activation"
        h = h * ((h*h)> theta)
    rec = T.sum((x - (T.dot(h, W.T)))**2, axis=1)
    cost = T.mean(rec)
    params = (W, )
    grads = T.grad(cost, params)
    return params, cost, grads, x
 

def test_zbae(hidden, indim, epochs, lr, momentum, btsz, batches,
        activ='TRec', theta=1., version="rotations"):
    """
    Test Zero bias AE on rotations.
    """
    if version is "rotations":
        print "Generating rotation data ..."
        data = rotations(btsz*batches, indim)
    else:
        print "Generating shift data ..."
        data = shifts(btsz*batches, indim)
    
    print "Building model ..."
    Winit = np.asarray(0.1 * np.random.standard_normal((data.shape[1], hidden)), dtype=theano.config.floatX)
    params, cost, grads, x = zbae(Winit, activ=activ, theta=theta)
    learner = {"lr": lr, "momentum": momentum}
    updates = momntm(params, grads, **learner)
    train = theano.function([x], cost, updates=updates, allow_input_downcast=True)
    # get data
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            cost += btsz*train(data[mbi*btsz:(mbi+1)*btsz])
        print epoch, cost
    return params


def initweight(variant="normal", **kwargs):
    """
    Init weights.
    """
    if variant is "normal":
        pass
    elif variant is "sparse":
        pass
    return


def rotations(samples, dims, dist=1., maxangle=30.):
    """
    Rotated dots for learning log-polar filters.
    """
    import scipy.ndimage
    tmps= np.asarray(np.random.randn(samples,4*dims*dims), dtype=np.float32)
    seq = np.asarray(np.zeros((samples, 2*dims*dims)), dtype=np.float32)
    for j, img in enumerate(tmps):
        _angle = np.random.vonmises(0.0, dist)/np.pi * maxangle
        tmp = scipy.ndimage.interpolation.rotate(img.reshape(2*dims, 2*dims),
            angle=_angle, reshape=False, mode='wrap')
        seq[j,:dims*dims] = tmp[dims/2:dims+dims/2,dims/2:dims+dims/2].ravel()
        _angle = np.random.vonmises(0.0, dist)/np.pi * maxangle
        tmp = scipy.ndimage.interpolation.rotate(img.reshape(2*dims, 2*dims),
            angle=_angle, reshape=False, mode='wrap')
        seq[j,dims*dims:] = tmp[dims/2:dims+dims/2,dims/2:dims+dims/2].ravel()
    return seq


def shifts(samples, dims, shft=3):
    """
    Produce shifted dots.
    """
    import scipy.ndimage
    shift = np.random.randn(samples,2*dims*dims)
    for j, img in enumerate(shift):
        _shift = np.random.randint(-shft, shft+1, 2)
        shift[j,dims*dims:] = scipy.ndimage.interpolation.shift(shift[j, :dims*dims].reshape(dims, dims), shift=_shift, mode='wrap').ravel()
    return shift


def momntm(params, grads, **kwargs):
    """
    Optimizer: SGD with momentum.
    """
    print "OPTIMIZER: SGD+Momentum"
    lr = kwargs['lr']
    momentum = kwargs['momentum']
    _moments = []
    for p in params:
        p_mom = theano.shared(np.zeros(p.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        _moments.append(p_mom)

    updates = OrderedDict()
    for grad_i, mom_i in zip(grads, _moments):
        updates[mom_i] =  momentum*mom_i + lr*grad_i

    for param_i, mom_i in zip(params, _moments):
            updates[param_i] = param_i - updates[mom_i]
    return updates
