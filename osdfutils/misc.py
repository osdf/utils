"""

"""


import numpy as np
import scipy as sp
import scipy.linalg as la


def sigmoid(x):
    return (1 + np.tanh(x/2.))/2.


def Dsigmoid(y):
    """
    Given y = sigmoid(x),
    what is dy/dx in terms of
    y!
    """
    return y*(1-y)


def Dtanh(y):
    """
    Given y = tanh(x),
    what is dy/dx in terms of
    y!
    """
    return 1 - y**2


def logcosh(y):
    """
    Smooth L1 penalty, log(cosh(y)).

    First derivative is tanh.
    """
    return np.log(np.cosh(y))


def sqrtsqr(y, eps=1e-8):
    """
    Soft L1 activation.
    """
    return sp.sqrt(eps + y**2)


def Dsqrtsqr(y, eps=1e-8):
    """
    First derivative of sqrtsqr(y)
    wrt y.
    """
    tmp = 1./sp.sqrt(eps + y**2)
    return y * tmp


Dtable = {
        sigmoid: Dsigmoid,
        np.tanh: Dtanh,
        logcosh: np.tanh,
        sqrtsqr: Dsqrtsqr
        }


def logsumexp(array, axis):
    """
    Compute log of (sum of exps) 
    along _axis_ in _array_ in a 
    stable way.
    """
    axis_max = np.max(array, axis)[:, np.newaxis]
    return axis_max + np.log(np.sum(np.exp(array-axis_max), axis))[:, np.newaxis]


def one2K(classes):
    """
    Given a numeric encoding of
    class membership in _classes_, 
    produce a 1-of-K encoding.
    """
    c = classes.max() + 1
    targets = np.zeros((len(classes), c))
    targets[classes] = 1
    return targets


def K2one(onehotcoding):
    return np.argmax(onehotcoding, axis=1)


def load_mnist():
    """
    Get standard MNIST: training, validation, testing.
    File 'mnist.pkl.gz' must be in path.
    Download it from ...
    """
    import gzip, cPickle
    print "Reading mnist.pkl.gz ..."
    f = gzip.open('mnist.pkl.gz','rb')
    trainset, valset, testset = cPickle.load(f)
    return trainset, valset, testset


def _scale_01(arr, eps=1e-10):
    """
    scale arr between [0,1].
    useful for gray images to be produced with PIL
    """
    newarr = arr.copy()
    newarr -= newarr.min()
    newarr *= 1.0/(newarr.max() + eps)
    return newarr


def receptive(array, ind):
    """
    Visualize receptive fields.
    """
    try:
        import Image as img 
    except:
        import PIL as img
    #
    sz = array.size
    fields = array.reshape(ind, sz/ind).T
    tiles = int(np.sqrt(sz/ind))
    notsquare = np.ceil( ((sz/ind) - (tiles**2))/(1.0 * tiles) )
    shape = int(np.sqrt(ind))
    pixelsy = (tiles + notsquare) * shape + (tiles + notsquare) + 1
    pixelsx = tiles * shape + tiles + 1
    # complete tiling
    tiling = np.zeros((pixelsy, pixelsx), dtype = 'uint8')
    for row in xrange(int(tiles + notsquare)):
        for col in xrange(tiles):
            if (col+row*tiles) < fields.shape[0]:
                tile = fields[col + row * tiles].reshape(shape, shape)
                tile = _scale_01(tile) * 255
                tiling[shape * row + row + 1:shape * (row+1) + row + 1,\
                        shape * col + col + 1:shape * (col+1) + col + 1] = tile
    return img.fromarray(tiling)


def dn(data, eps=1e-8):
    """
    Devisive normalization.

    _data_ consists of rows.
    """
    norm = np.sqrt(np.sum(data**2, axis=1) + eps)
    data /= np.atleast_2d(norm).T
    return


def cn(data):
    """
    Coordinate wise normalization.
    
    Every coordinate ('feature') becomes
    zero-mean and unit-std normalized.
    Changes data *inplace*.
    """
    mu = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data -= mu
    data /= std
    return 
