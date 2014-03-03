"""
Some theano offsprings.
"""


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
    h = T.dot(x, W)
    if activ is "TRec":
        print "Using TRec as activation"
        h = h * (h > theta)
    else:
        print "Using TLin as activation"
        h = h * ((h*h)> theta)
    rec = T.sum(x - T.dot(h, W.T), axis=1)
    cost = T.mean(rec)
    params = [W]
    grads = T.grad(cost, W)
    return params, cost, grads
 

def test_zbae(hidden, indim, epochs, lr, momentum, btsz, batches,
        activ='TRec', theta=1.):
    """
    Test Zero bias AE on rotations.
    """
    Winit = np.asarray(np.random.standard_normal((indim, hidden)), dtype=theano.config.floatX)
    params, cost, grads = zbae(Winit, activ=activ, theta=theta)
    updates = momentum(params, grads)
    train = theano.function([x], updates=updates, allow_input_downcast=True)
    # get data
    data = rotations(btsz*batches, indim)
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            cost += train(data[mbi*btsz:(mbi+1)*btsz])
        print epoch, cost


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


def shifts(samples, dims, shift=3):
    """
    Produce shifted dots.
    """
    import scipy.ndimage
    ins = numpy.random.randn(samples,dims*dims)
    outs = numpy.zeros((samples, dims*dims))
    for j, img in enumerate(ins):
        _shift = numpy.random.randint(-shift, shift+1, 2)
        outs[j,:] = scipy.ndimage.interpolation.shift(ins[j].reshape(dims, dims), shift=_shift, mode='wrap').ravel()
    return ins, outs


def momentum(params, grads, **kwargs):
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
