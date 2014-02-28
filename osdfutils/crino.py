"""
Some theano offsprings.
"""


def skmeans():
    """
    synchronous k-means.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX,
        borrow=True, name='W')
    sprod = T.dot(x, W)

    cost = T.sum((X - np.dot(?, W.T))**2)
    grads = T.grad(cost, W)


def sae():
    """
    synchronous autoencoder.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX,
        borrow=True, name='W')
    h = T.dot(x, W)
    sh = h*h
    rec = T.sum(x - T.dot(sh*h, W.T), axis=1)
    cost = T.mean(rec)
    grads = T.grad(cost, W)

def zbae(activ='TRec', theta):
    """
    Zero bias autoencoder.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX,
        borrow=True, name='W')
    h = T.dot(x, W)
    if activ is "TRec":
        print "Using TRec as activation"
        h = h * (h > theta)
    else:
        print "Using TLin as activation"
        h = h * ((h*h)> theta)
    rec = T.sum(x - T.dot(h, W.T), axis=1)
    cost = T.mean(rec)
    grads = T.grad(cost, W)


def rotations(samples, dims, dist=1., maxangle=30.):
    """
    Rotated dots for learning log-polar filters.
    """
    import scipy.ndimage
    tmps= numpy.random.randn(samples,4*dims*dims)
    ins = numpy.zeros((samples, dims*dims))
    outs = numpy.zeros((samples, dims*dims))
    for j, img in enumerate(tmps):
        _angle = numpy.random.vonmises(0.0, dist)/numpy.pi * maxangle
        tmp = scipy.ndimage.interpolation.rotate(img.reshape(2*dims, 2*dims),
            angle=_angle, reshape=False, mode='wrap')
        outs[j,:] = tmp[dims/2:dims+dims/2,dims/2:dims+dims/2].ravel()
        _angle = numpy.random.vonmises(0.0, dist)/numpy.pi * maxangle
        tmp = scipy.ndimage.interpolation.rotate(img.reshape(2*dims, 2*dims),
            angle=_angle, reshape=False, mode='wrap')
        ins[j,:] = tmp[dims/2:dims+dims/2,dims/2:dims+dims/2].ravel()
    return ins, outs
