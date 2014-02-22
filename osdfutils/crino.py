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
