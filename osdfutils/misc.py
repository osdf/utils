"""

"""


import numpy as np
import scipy as sp


def identy(x):
    return x


def Didenty(x):
    return 1


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
        identy: Didenty
        ,sigmoid: Dsigmoid
        ,np.tanh: Dtanh
        ,logcosh: np.tanh
        ,sqrtsqr: Dsqrtsqr
        }


def logsumexp(array, axis):
    """
    Compute log of (sum of exps) 
    along _axis_ in _array_ in a 
    stable way. _array_ is in the log domain.

    If _axis_ is not zero, caller must transform
    result in suitable shape.
    """
    arr = np.rollaxis(array, axis)
    axis_max = np.max(arr, axis=0)
    return axis_max + np.log(np.sum(np.exp(arr - axis_max), axis=0))
 

def norm_logprob(logprobs, axis):
    """Given log probabilities _logprobs_, convert to equivalent
    discrete probility distribution along _axis_.

    If _axis_ is not 0, caller must transpose accordingly
    to get original shape of _logprobs_ back.
    """
    logprobs = np.rollaxis(logprobs, axis)
    axis_max = np.max(logprobs, axis=0)
    tmp = logprobs - axis_max
    tmp = np.exp(tmp) + np.finfo(np.float).tiny
    return tmp/np.sum(tmp, axis=0)


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

def visualize(array, rsz, shape_r=None, xtiles=None, fill=0):
    """Visualize flattened bitmaps.

    _array_ is supposed to be a 1d array that
    holds the bitmaps (of size _rsz_ each)
    sequentially. _rsz_ must be a square number
    if _shape_r_ is None.

    Specifiy the number of rows with _xtiles_.
    If not specified, the layout is approximately
    square. _fill_ defines the pixel border between
    patches (default is black (==0)).
    """
    try:
        import Image as img
    except:
        import PIL as img
    sz = array.size
    fields = array.reshape(sz/rsz, rsz)
    
    # tiles per axes
    xtiles = xtiles if xtiles else int(np.sqrt(sz/rsz))
    ytiles = int(np.ceil(sz/rsz/(1.*xtiles)))
    if shape_r is None:
        shape_r = int(np.sqrt(rsz))
    shape_c = int(rsz/shape_r)
    
    # take care of extra pixels for borders
    pixelsy = ytiles * shape_r + ytiles + 1
    pixelsx = xtiles * shape_c + xtiles + 1
    # the tiling has this shape and _fill_ background
    tiling = fill*np.ones((pixelsy, pixelsx), dtype = 'uint8')
    
    for row in xrange(ytiles):
        for col in xrange(xtiles):
            if (col+row*xtiles) < fields.shape[0]:
                tile = fields[col + row * xtiles].reshape(shape_r, shape_c)
                tile = _scale_01(tile) * 255
                tiling[shape_r * row + row + 1:shape_r * (row+1) + row + 1, shape_c * col + col + 1:shape_c * (col+1) + col + 1] = tile
    return img.fromarray(tiling)

def hinton(array, sqr_sz = 9):
    """A hinton diagram without matplotlib.
    Code definetely has potential for improvement.

    _array_ is the one to visualize. _sqr_sz_ is
    the length of a square. Bigger -> more details.

    See https://gist.github.com/292018
    """
    try:
        import Image as img
    except:
        import PIL as img

    dx, dy = array.shape
    W = 2**np.ceil(np.log(np.abs(array).max())/np.log(2))
    # take care of extra pixels for borders
    pixelsy = dy * sqr_sz + dy + 1
    pixelsx = dx * sqr_sz + dx + 1
    tiling = 128*np.ones((pixelsx, pixelsy), dtype = 'uint8')
    for (x,y), w in np.ndenumerate(array):
        xpos = x * sqr_sz + x + 1 + int(sqr_sz/2 + 1)
        ypos = y * sqr_sz + y + 1 + int(sqr_sz/2 + 1)
        dw = int(np.abs(w)/W * sqr_sz)/2 + 1
        cw = (w > 0) * 255
        tiling[xpos - dw:xpos + dw, ypos - dw:ypos+dw] = cw
    return img.fromarray(tiling)


def dn(data, eps=1e-8):
    """
    Devisive normalization.

    _data_ consists of rows.
    Operation is *inplace*.
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


def check_grad(f, fprime, x0, args, eps=1e-8, verbose=False):
    """
    Numerically check the gradient, using
    """
    # computed gradient at x0
    grad = fprime(x0, **args)

    # space for the numeric gradient
    ngrad = np.zeros(grad.shape)
    perturb = np.zeros(grad.shape)

    if verbose:
        print "Total number of calls to f: 2*%d=%d"% (x0.shape[0], 2*x0.shape[0])
    for i in xrange(x0.shape[0]):
        perturb[i] = eps

        f1 = f(x0 + perturb, **args)
        f2 = f(x0 - perturb, **args)

        ngrad[i] = (f1 - f2)/(2*eps)

        # undo eps
        perturb[i] = 0.
    norm_diff = np.sqrt(np.sum((grad-ngrad)**2))
    norm_sum = np.sqrt(np.sum((grad+ngrad)**2))

    if verbose:
        print "Gradient: ", grad
        print "Numerical Approximation: ", ngrad
        print "Norm difference:", norm_diff
        print "Relative norm difference:", norm_diff/norm_sum

    return norm_diff/norm_sum
