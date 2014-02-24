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


def sae():
    """
    synchronous autoencoder.
    """
    pass


def zbae():
    """
    Zero bias autoencoder.
    """
    x = T.matrix('x')
    W = theano.shared(np.asarray(Winit, dtype=theano.config.floatX,
        borrow=True, name='W')
    h = T.dot(x, W)
    h *= (h>theta)
    rec = T.sum(x - T.dot(h, W.T), axis=1)
    cost = T.mean(rec)
    grads = T.grad(cost, W)
