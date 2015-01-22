"""
Some theano offspring.
"""
"""
The MIT License (MIT)

Copyright (c) 2011-2015 Christian Osendorfer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""" 
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv as Tconv
from misc import logsumexp



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


def sae(shape, activ, lmbd, init='uniform'):
    """
    Synchronous autoencoder for motion, see 
    Learning to encode motion using spatio-temporal synchrony
    Konda,K., Memisevic, R., Michalski, V.
    eq. 20

    _shape_ is the shape of the weight matrix.
    _activ_ is the hidden activation fct.
    _lmbd_ is the weighting of the contractive penalty.
    """
    x = T.matrix('x')
    wi = initweight(shape, variant=init)
    W = theano.shared(wi, borrow=True, name='W')
    #_b = np.zeros((shape[0],), dtype=theano.config.floatX)
    #b1 = theano.shared(value=_b, borrow=True)

    h = T.dot(x, W)
    sh = activ(h*h)
    
    cae = T.grad(T.sum(T.mean(sh, axis=0)), h)
    cae = T.sum( T.mean(cae*cae, axis=0) * T.sum(T.sqr(W), axis=0) )
    
    _b = np.zeros((shape[0],), dtype=theano.config.floatX)
    b1 = theano.shared(value=_b, borrow=True)
    xhat = T.dot(sh*h, W.T) + b1
    
    rec = T.sum( (x - xhat)**2, axis=1 )
    cost = T.mean(rec) + lmbd * cae

    params = (W, b1)
    grads = T.grad(cost, params)
    return params, cost, grads, x


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
    inits = {"std": 0.1, "n": data.shape[1], "m": hidden}
    Winit = initweight(variant="normal", **inits) 
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


def sh(x, theta):
    """Simple shrinkage function.
    """
    return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)


def lode(config, activ, tied=True):
    """
    Returns x, params, cost, grads
    """
    print "[LODE]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    Qinit = config['Q']

    x = T.matrix('x')
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')
    
    # Q for ??
    _Q = initweight(**Qinit)
    # normalize atoms of dictionary (== rows) to length 1
    _Qdiag = np.diag(_Q)
    _Qrest = _Q - np.diag(_Qdiag)
    _Qrest = (-1)*np.sign(_Qrest)*_Qrest
    _Q = _Qrest + np.diag(0*_Qdiag)
    _q = np.sqrt(np.sum(_Q * _Q, axis=1, keepdims=True))
    _Q /= _q
    Q = theano.shared(value=np.asarray(_Q, dtype=theano.config.floatX),
            borrow=True, name='Q')

    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    if tied:
        W = D.T
        params = [D, Q, L]
    else:
        _W = initweight(**Dinit).T
        _w = np.sqrt(np.sum(_W * _W, axis=0, keepdims=True))
        _W /= _w
        W = theano.shared(value=np.asarray(_W, dtype=theano.config.floatX),
            borrow=True, name='W')
        params = [D, W, Q, L]

    _theta = np.random.randn(_D.shape[0],)
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX), 
            borrow=True, name="theta")
    params.append(theta)

    b = T.dot(x, W)
    z = activ(b + theta)
    for i in range(layers):
        b = T.dot(x, W) + T.dot(z, Q)
        z = (1-L)*z + L*activ(b + theta)

    rec = T.dot(z, D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    cost = cost + sp_lmbd * sparsity
    return x, params, cost, rec, z


def lruku(config, activ, tied=True):
    """
    Returns x, params, cost, grads
    """
    print "[LODE]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    Qinit = config['Q']

    x = T.matrix('x')
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')
    
    # Q for ??
    _Q = initweight(**Qinit)
    # normalize atoms of dictionary (== rows) to length 1
    _Qdiag = np.diag(_Q)
    _Qrest = _Q - np.diag(_Qdiag)
    _Qrest = (-1)*np.sign(_Qrest)*_Qrest
    _Q = _Qrest + np.diag(0*_Qdiag)
    _q = np.sqrt(np.sum(_Q * _Q, axis=1, keepdims=True))
    _Q /= _q
    Q = theano.shared(value=np.asarray(_Q, dtype=theano.config.floatX),
            borrow=True, name='Q')

    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    if tied:
        W = D.T
        params = [D, Q, L]
    else:
        _W = initweight(**Dinit).T
        _w = np.sqrt(np.sum(_W * _W, axis=0, keepdims=True))
        _W /= _w
        W = theano.shared(value=np.asarray(_W, dtype=theano.config.floatX),
            borrow=True, name='W')
        params = [D, W, Q, L]

 
    _theta = np.random.randn(_D.shape[0],)
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX), 
            borrow=True, name="theta")
    params.append(theta)


    b = T.dot(x, W)
    z = activ(b + theta)
    for i in range(layers):
        b = T.dot(x, W) + T.dot(z, Q)
        zprime = (1-L)*z + L*activ(b+theta)
        b1 = activ(b + theta) - z
        b = T.dot(x, W) + T.dot(zprime, Q)
        b2 = activ(b + theta) - zprime
        z = z + L/2*b1 + L/2*b2

    rec = T.dot(z, D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    cost = cost + sp_lmbd * sparsity
    return x, params, cost, rec, z


def lodeconv(config, activ, tied=True):
    """
    Returns x, params, cost, grads
    """
    print "[LODE-CONV]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    Qinit = config['Q']
    Winit = config['W']
    imshape = config['imshape']
    x = T.matrix('x')
    x = x.reshape(imshape)
    # D for reconstruction convolutional dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== filters) to length 1
    _d = np.sqrt(np.sum(_D**2, axis=(2, 3), keepdims=True))
    _D /= _d
    print "D", _D.shape
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')
    
    # Q for ??
    _Q = initweight(**Qinit)
    # normalize atoms of dictionary to length 1
    # special case here: 1x1 filters over feature maps
    q1, q2, q3, q4 = Qinit['shape']
    assert q3 == 1, "1x1 convolution for Q!"
    assert q4 == 1, "1x1 convolution for Q!"
    _Q = _Q.reshape(q1, q2)
    # make sure Q has only negative elements on non diagonals.
    _Qdiag = np.diag(_Q)
    _Qrest = _Q - np.diag(_Qdiag)
    _Qrest = (-1)*np.sign(_Qrest)*_Qrest
    _Q = _Qrest + np.diag(0*_Qdiag+1)
    _q = np.sqrt(np.sum(_Q**2, axis=1, keepdims=True))
    _Q /= _q
    _Q = _Q.reshape(q1, q2,1, 1)
    print "Q", _Q.shape
    Q = theano.shared(value=np.asarray(_Q, dtype=theano.config.floatX),
            borrow=True, name='Q')

    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    params = [D, L]
    if tied:
        W = D.T
    else:
        _W = initweight(**Winit)
        _w = np.sqrt(np.sum(_W**2, axis=(2, 3), keepdims=True))
        _W /= _w
        print "W", _W.shape
        W = theano.shared(value=np.asarray(_W, dtype=theano.config.floatX),
            borrow=True, name='W')
        params.append(W)
    _theta = np.random.randn(Winit['shape'][0],)
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX), 
            borrow=True, name="theta")
    params.append(theta)

    w1, w2, w3, w4 = Winit['shape']
    zimshape = (imshape[0], w1, imshape[2] - w3 + 1, imshape[3] - w4 + 1) 
    
    # first iteration of ode.
    b = Tconv.conv2d(input=x, filters=W, filter_shape=Winit['shape'],
            border_mode="valid", image_shape=imshape)
    z = activ(b + theta.dimshuffle('x', 0, 'x', 'x'))
    for i in range(layers):
        _b = Tconv.conv2d(input=x, filters=W, filter_shape=Winit['shape'],
            border_mode="valid", image_shape=imshape)
        b = _b + Tconv.conv2d(input=z, filters=Q, filter_shape=Qinit['shape'],
                border_mode="valid", image_shape=zimshape)
        z = (1-L)*z + L*activ(b + theta.dimshuffle('x', 0, 'x', 'x'))

    rec = Tconv.conv2d(input=z, filters=D, filter_shape=Dinit['shape'],
            border_mode="full", image_shape=zimshape)
    cost = T.mean(T.sum((x - rec)**2, axis=(2,3)))
    sparsity = T.mean(T.sum(T.abs_(z), axis=(1,2,3)))
    cost = cost + sp_lmbd * sparsity
    params.append(Q)
    return x, params, cost, rec, z


def mfnw(config, activ, tied=True):
    """
    Returns x, params, cost, grads
    """
    print "[MFNW]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    Qinit = config['Q']

    x = T.matrix('x')
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')
    
    # Q for ??
    _Q = initweight(**Qinit)
    # normalize atoms of dictionary (== rows) to length 1
    _Qdiag = np.diag(_Q)
    _Qrest = _Q - np.diag(_Qdiag)
    _Qrest = (-1)*np.sign(_Qrest)*_Qrest
    _Q = _Qrest + np.diag(0*_Qdiag)
    _q = np.sqrt(np.sum(_Q * _Q, axis=1, keepdims=True))
    _Q /= _q
    Q = theano.shared(value=np.asarray(_Q, dtype=theano.config.floatX),
            borrow=True, name='Q')

    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    if tied:
        W = D.T
        params = [D, Q, L]
    else:
        _W = initweight(**Dinit).T
        _w = np.sqrt(np.sum(_W * _W, axis=0, keepdims=True))
        _W /= _w
        W = theano.shared(value=np.asarray(_W, dtype=theano.config.floatX),
            borrow=True, name='W')
        params = [D, W, Q, L]

    _theta = np.random.randn(_D.shape[0],)
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX), 
            borrow=True, name="theta")
    params.append(theta)

    b = T.dot(x, W)
    z = activ(b + theta)
    for i in range(layers):
        b = T.dot(x, W) + T.dot(z, Q)
        z = (1-L)*z + L*activ(b + theta)

    rec = T.dot(z, D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    cost = cost + sp_lmbd * sparsity
    return x, params, cost, rec, z


def drsae(config, activ, tied=True):
    """
    Discriminative Recurrent Sparse AE.
    Returns x, params, cost, grads
    """
    print "[LODE]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    Qinit = config['Q']

    x = T.matrix('x')
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')
    
    # Q for ??
    _Q = initweight(**Qinit)
    # normalize atoms of dictionary (== rows) to length 1
    _Qdiag = np.diag(_Q)
    _Qrest = _Q - np.diag(_Qdiag)
    _Qrest = (-1)*np.sign(_Qrest)*_Qrest
    _Q = _Qrest + np.diag(0*_Qdiag)
    _q = np.sqrt(np.sum(_Q * _Q, axis=1, keepdims=True))
    _Q /= _q
    Q = theano.shared(value=np.asarray(_Q, dtype=theano.config.floatX),
            borrow=True, name='Q')

    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    if tied:
        W = D.T
        params = [D, Q, L]
    else:
        _W = initweight(**Dinit).T
        _w = np.sqrt(np.sum(_W * _W, axis=0, keepdims=True))
        _W /= _w
        W = theano.shared(value=np.asarray(_W, dtype=theano.config.floatX),
            borrow=True, name='W')
        params = [D, W, Q, L]

    _theta = np.random.randn(_D.shape[0],)
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX), 
            borrow=True, name="theta")
    params.append(theta)

    b = T.dot(x, W)
    z = activ(b + theta)
    for i in range(layers):
        b = T.dot(x, W) + T.dot(z, Q)
        z = (1-L)*z + L*activ(b + theta)

    rec = T.dot(z, D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    cost = cost + sp_lmbd * sparsity
    return x, params, cost, rec, z


def lista(config, shrinkage):
    """Learned ISTA by Gregor/Lecun.

    Returns x, params, cost, grads
    """
    print "[LISTA]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    
    x = T.matrix('x')
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')

    _S = np.eye(_D.shape[0]) - 1./L * np.dot(_D ,_D.T)
    S = theano.shared(value=np.asarray(_S, dtype=theano.config.floatX),
            borrow=True, name='S')
    
    _theta = np.abs(np.random.randn(_S.shape[0],))
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX),
            borrow=True, name="theta")
    
    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    params = (D, S, theta, L)
    
    b = T.dot(x, D.T)
    z = shrinkage(b, theta)
    for i in range(layers):
        c = b + T.dot(z, S)
        z = shrinkage(c, theta)


    rec = T.dot(z, L * D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    cost = cost + sp_lmbd * sparsity
    return x, params, cost, rec, z

 
def lcod(config, shrinkage):
    """Learned CoD by Gregor/Lecun.

    Returns x, params, cost, grads
    """
    print "[LCoD]"
    layers = config['layers']
    sp_lmbd = config['lambda']
    L = config['L']
    Dinit = config['D']
    
    x = T.matrix('x')
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')

    _S = np.eye(_D.shape[0]) - 1./L * np.dot(_D ,_D.T)
    S = theano.shared(value=np.asarray(_S, dtype=theano.config.floatX),
            borrow=True, name='S')
    
    _theta = np.abs(np.random.randn(_S.shape[0],))
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX),
            borrow=True, name="theta")
    
    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")

    params = (D, S, theta, L)
    
    b = T.dot(x, D.T)
    z = 0
    for i in range(layers):
        znew = shrinkage(b, theta)
        e = T.abs_(znew - z)
        k = T.argmax(e, axis=1)
        e = znew - z
        e = e[T.arange(e.shape[0]), k]
        Sjk = S[:, k].T
        b = b + Sjk* e.dimshuffle(0, 'x')
        z = znew 
    z = shrinkage(b, theta)

    rec = T.dot(z, L * D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    cost = cost + sp_lmbd * sparsity
    return x, params, cost, rec, z


#def bcofb(config, shrinkage):
#    """Learned CoD by Gregor/Lecun.
#
#    Returns x, params, cost, grads
#    """
#    print "[BCoFB]"
#    layers = config['layers']
#    sp_lmbd = config['lambda']
#    L = config['L']
#    Dinit = config['D']
#    
#    x = T.matrix('x')
#    
#    # D for dictionary
#    _D = initweight(**Dinit)
#    # normalize atoms of dictionary (== rows) to length 1
#    _d = np.sqrt(np.sum(_D * _D, axis=1))
#    _D /= np.atleast_2d(_d).T
#    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
#            borrow=True, name='D')
#
#    _S = np.eye(_D.shape[0]) - 1./L * np.dot(_D ,_D.T)
#    S = theano.shared(value=np.asarray(_S, dtype=theano.config.floatX),
#            borrow=True, name='S')
#    
#    _theta = np.abs(np.random.randn(_S.shape[0],))
#    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX),
#            borrow=True, name="theta")
#    
#    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
#            borrow=True, name="L")
#
#    params = (D, S, theta, L)
#
#    # local defs
#    def pnorms(idx1, idx2, x):
#        return T.sum(x[:, idx1:idx2]**2, axis=1)
#
#    def subdot(idx, e , idx1, idx2, S):
#        return T.dot(S[:, idx1[idx]:idx2[idx]], e[idx1[idx]:idx2[idx]].T).T
#    
#    def subtens(z,y,g,lower,upper): 
#        return T.set_subtensor( z[lower[g]:upper[g]] , y[lower[g]:upper[g]])
#    b = T.dot(x, D.T)
#    z = 0
#    for i in range(layers):
#        y = None
#        e = y - z
#        
#        norms, _updt = theano.map(pnorms, sequences[lower, upper], non_sequences=[e])
#        g = T.argmax(norms.T, axis=1)
#
#        delta_b, _updt = theano.map(subdot, sequences=[g, e], non_sequences=[lower, upper, S])
#        b = b + delta_b
#
#        z, _updt = theano.map(subtens, sequences=[z, y, g], non_sequences=[lower, upper])
#    z = shrinkage(b, theta)
#
#    rec = T.dot(z, L * D)
#    cost = T.mean(T.sum((x - rec)**2, axis=1))
#    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
#    cost = cost + sp_lmbd * sparsity
#    return x, params, cost, rec, z


def lconvista(config, shrinkage):
    """Learned Convolutional ISTA   

    See github.com/msoelch/FastApproximations.
    """
    print "[LConvISTA]"

    layers = config['layers']
    btsz = config['btsz']
    lmbd = config['lambda']

    Dinit = config['D'] 
    _D = initweight(**Dinit)
    _d = np.sqrt(np.sum(_D * _D, axis=1, keepdims=True))
    _D /= _d
    _D = _D.reshape(Dinit["tensor"])
    D = theano.shared(value=np.asarray(_D, 
            dtype=theano.config.floatX),borrow=True,name='D')
    _theta = config['theta']
    theta = theano.shared(value=np.asarray(_theta, 
            dtype=theano.config.floatX),borrow=True,name="theta")
    _L = config['L']
    L = theano.shared(value=np.asarray(_L, 
            dtype=theano.config.floatX),borrow=True,name="L")

    params = [D, theta, L]

    #filter shape information for speed up of convolution
    fs1 = _D.shape
    fs2 = (fs1[1],fs1[0],fs1[2],fs1[3])

    imshape = config['imshape']
    x = T.matrix('x')
    _x = x.reshape(imshape)
    
    z = T.zeros(config['zshape'], dtype=theano.config.floatX)

    # The combination of for loop and 'hand calculated' gradient was tested
    # on CPU with 2 layers and 16 filters as well as 5 layers and 49 filters.
    # Note though that T.grad catches up with increasing parameters.
    # Hand calculated grad is preferred due to higher flexibility. 
    for i in range(layers):
        deconv = Tconv.conv2d(z, D, border_mode='valid', 
                image_shape=config['zshape'], filter_shape=fs1) 
        gradZ = Tconv.conv2d(deconv  - _x, D[:,:,::-1,::-1].dimshuffle(1,0,2,3),
                border_mode='full', image_shape=config['imshape'],
                filter_shape=fs2)
        gradZ = gradZ/btsz
        z = shrinkage(z - 1/L * gradZ, theta)


    def rec_error(_x, _z, D):
        # Calculates the reconstruction rec_i = sum_j Z_ij * D_j
        # and the corresponding (mean) square reconstruction error
        # rec_error = (X_i - rec_i) ** 2

        rec = Tconv.conv2d(_z, D, border_mode='valid', image_shape=config['zshape'],
                filter_shape=_D.shape) 
        error = 0.5*T.mean( ((x - rec)**2).sum(axis=-1).sum(axis=-1))
        return error, rec


    sparsity = T.mean(T.sum(T.sum(T.abs_(z),axis=-1),axis=-1))
    rec_err, rec = rec_error(_x, z, D)    

    cost = rec_err + lmbd*sparsity
    return x, params, z, rec, rec_err, cost, sparsity


def initweight(shape, variant="normal", **kwargs):
    """
    Init weights.
    """
    if variant is "normal":
        if "std" in kwargs:
            std = kwargs["std"]
        else:
            std = 0.1
        weights = np.asarray(np.random.normal(loc=0, scale=std, size=shape),
                dtype=theano.config.floatX)
    elif variant is "uniform":
        if len(shape) == 2:
            units = shape[0]*shape[1]
        elif len(shape) == 4:
            units = np.prod(shape[1:])
            _tmp = units * np.prod(shape[2:])
            units = units + _tmp
        else:
            assert False, "Shape in initweight is difficult to handle."
        bound = 4*np.sqrt(6./units)
        weights = np.asarray(np.random.uniform(low=-bound, high=bound, size=shape),
                dtype=theano.config.floatX)
    elif variant is "sparse":
        sparsity = kwargs["sparsity"]
        weights = np.zeroes(shape, dtype=theano.config.floatX)
        for w in weights:
            w[random.sample(xrange(n), sparsity)] = np.random.randn(sparsity)
        weights = weights.T
    else:
        assert False, 'Problem in initweight.'

    return weights


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


def momntm(params, grads, settings, **kwargs):
    """
    Optimizer: SGD with momentum.
    """
    lr = settings['lr']
    momentum = settings['momentum']
    print "[MOMNTM] lr: {0}; momentum: {1}".format(lr, momentum)

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


def norm_updt(params, updates, todo):
    """Normalize updates wrt length.
    """
    for p in params:
        if p.name in todo:
            axis = todo[p.name]['axis']
            const = todo[p.name]['c']
            print "[NORM_UPDT] {0} normalized to {1} along axis {2}".format(p.name, const, axis)
            wl = T.sqrt(T.sum(T.square(updates[p]), axis=axis, keepdims=True) + 1e-6)
            updates[p] = const * updates[p]/wl
    return updates


def max_updt(params, updates, todo):
    """Normalize updates wrt to minimum value.
    """
    for p in params:
        if p.name in todo:
            thresh = todo[p.name]['thresh']
            print "[MAX_UPDT] {0} at least {1}".format(p.name, thresh)
            updates[p] = T.maximum(thresh, updates[p])
    return updates


def min_updt(params, updates, todo):
    """Normalize updates wrt to maximum value.
    """
    for p in params:
        if p.name in todo:
            thresh = todo[p.name]['thresh']
            print "[MIN_UPDT] {0} at most {1}".format(p.name, thresh)
            updates[p] = T.minimum(thresh, updates[p])
    return updates


def mlp(config, params, im):
    """
    A mlp acting as an encoding or decoding layer --
    This depends on the loss that is specified
    in the _config_ dictionary.
    """
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']

    assert len(shapes) == len(activs),\
            "[MLP -- {0}]: One layer, One activation.".format(tag)
    
    print "[MLP -- {0}]: MLP with {1} layers.".format(tag, len(shapes))


    if 'tied' in config:
        tied = config['tied']
    else:
        tied = {}

    inpt = im[config['inpt']]

    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[MLP -- {0}]: Input is a 1-list, taking first element.".format(tag)
            inpt = inpt[0]

    _tmp_name = config['inpt']
    for i, (shape, act) in enumerate(zip(shapes, activs)):
        # fully connected
        if i in tied:
            _tied = False
            for p in params:
                if tied[i] == p.name:
                    print "Tying layer {0} in {1} with {2}".format(i, tag, p.name)
                    _w = p.T
                    _w = _w[:shape[0], :shape[1]]
                    _tied = True
            assert _tied,\
                    "[MLP -- {0}]: Tying was set for layer {1}, but unfulfilled!".format(tag, i)
        else:
            _tmp = initweight(shape, variant=config["init"])
            _tmp_name = "{0}_w{1}".format(tag, i)
            _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
            params.append(_w)
        # bias
        _tmp = np.zeros((shape[1],), dtype=theano.config.floatX)
        _tmp_name = "{0}_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)
        inpt = act(T.dot(inpt, _w) + _b)
        _tmp_name = "{0}_layer{1}".format(tag, i)
        im[_tmp_name] = inpt

    config['otpt'] = _tmp_name


def pmlp(config, params, im):
    """
    A pmlp acting as an encoding or decoding layer --
    a 'pmlp' is a
    MLP with product interactions
    """
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']
    noises = config['noises']
    assert (len(shapes) == len(activs)) and (len(shapes) == len(noises)),\
            "[PMLP -- {0}]: One layer, One activation, One noise.".format(tag)

    rng = T.shared_randomstreams.RandomStreams()
 
    inpt = im[config['inpt']]
    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[PMLP -- {0}]: Input is a 1-list, taking first element.".format(tag)
            inpt = inpt[0]


    #TODO
    #should w1 and w2 be shared? -> make option
    _tmp_name = config['inpt']
    for i, (shape, act, noise) in enumerate(zip(shapes, activs, noises)):
        
        if noise[0] == "01":
            inpt1 = rng.binomial(size=inpt.shape, n=1, p=1.0-noise[1], dtype=theano.config.floatX) * inpt
            inpt2 = rng.binomial(size=inpt.shape, n=1, p=1.0-noise[1], dtype=theano.config.floatX) * inpt
        elif noise[0] == "gauss":
            inpt1 = rng.normal(size=inpt.shape, std=noise[1], dtype=theano.config.floatX) + inpt
            inpt2 = rng.normal(size=inpt.shape, std=noise[1], dtype=theano.config.floatX) + inpt
        else:
            assert False, "[PMLP -- {0}: Unknown noise process.".format(tag)
        
        _tmp = initweight(shape[:2], variant=config["init"])
        _tmp_name = "{0}_w1{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        im['normalize'][_tmp_name] = 0

        fac1 = T.dot(inpt1, _w)
        params.append(_w)

        _tmp = initweight(shape[:2], variant=config["init"])
        _tmp_name = "{0}_w2{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        im['normalize'][_tmp_name] = 0

        fac2 = T.dot(inpt2, _w)
        params.append(_w)

        prod = fac1 * fac2

        _tmp = initweight(shape[1:], variant=config["init"])
        _tmp_name = "{0}_w3{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[2],), dtype=theano.config.floatX)
        _tmp_name = "{0}_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)
        
        inpt = act(T.dot(prod, _w) + _b)
        _tmp_name = "{0}_layer{1}".format(tag, i)
        im[_tmp_name] = inpt
    config['otpt'] = _tmp_name


def conv(config, params, im):
    """
    Convolutional encoder.
    """
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']
    pools = config['pools']

    assert len(shapes) == len(activs),\
            "[CNN -- {0}]: One layer, One activation.".format(tag)
    assert len(shapes) == len(pools),\
            "[CNN -- {0}]: One layer, One Pool.".format(tag)

    imshape = config['imshape']

    print "[CNN -- {0}]: CNN with {1} layers, input image {2}.".format(tag, len(shapes), imshape)
    
    init = config["init"]

    inpt = im[config['inpt']]
    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[CNN -- {0}]: Input is a 1-list, taking first element.".format(tag)
            inpt = inpt[0]

    inpt = inpt.reshape(imshape)
    _tmp_name = config['inpt']
    for i, (shape, act, pool) in enumerate(zip(shapes, activs, pools)):
        assert imshape[1] == shape[1],\
            "[CNN -- {0}, L{1}]: Input and Shapes need to fit.".format(tag, i)

        if init == "normal":
            print "[CNN -- {0}, L{1}]: Init shape {2} via Gaussian.".format(tag, i, shape)
            _tmp = {"std": 0.1}
            _tmp = initweight(shape, variant=init, **_tmp) 
        else:
            print "[CNN -- {0}, L{1}]: Init shape {2} via Uniform.".format(tag, i, shape)
            fan_in = np.prod(shape[1:])
            fan_out = (shape[0] * np.prod(shape[2:]) / np.prod(pool))
            winit = np.sqrt(6. / (fan_in + fan_out))
            _tmp = np.asarray(np.random.uniform(low=-winit, high=winit,
                size=shape), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_w{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[0],), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)

        _conv = Tconv.conv2d(input=inpt, filters=_w, filter_shape=shape,
                image_shape=imshape)
        _tmp_name = "{0}_conv_layer{1}".format(tag, i)
        im[_tmp_name] = _conv

        _pool = downsample.max_pool_2d(input=_conv, ds=pool, ignore_border=True)
        _tmp_name = "{0}_pool_layer{1}".format(tag, i)
        im[_tmp_name] = _pool

        inpt = act(_pool + _b.dimshuffle('x', 0, 'x', 'x'))
        _tmp_name = "{0}_cnn_layer{1}".format(tag, i)
        im[_tmp_name] = inpt
        imshape = (imshape[0], shape[0], (imshape[2] - shape[2] + 1)//pool[0],\
                (imshape[3] - shape[3] + 1)//pool[1])
    im[_tmp_name] = im[_tmp_name].flatten(2)
    config['otpt'] = _tmp_name


def pconv(config, params, im):
    """
    Product Convolutional encoder.
    TODO.
    """
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']
    pools = config['pools']

    assert len(shapes) == len(activs),\
            "[PCNN -- {0}]: One layer, One activation.".format(tag)
    assert len(shapes) == len(pools),\
            "[PCNN -- {0}]: One layer, One Pool.".format(tag)

    imshape = config['imshape']

    print "[CNN -- {0}]: CNN with {1} layers, input image {2}.".format(tag, len(shapes), imshape)
    
    init = config["init"]

    inpt = im[config['inpt']]
    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[CNN -- {0}]: Input is a 1-list, taking first element.".format(tag)
            inpt = inpt[0]

    inpt = inpt.reshape(imshape)
    _tmp_name = config['inpt']
    for i, (shape, act, pool) in enumerate(zip(shapes, activs, pools)):
        assert imshape[1] == shape[1],\
            "[CNN -- {0}, L{1}]: Input and Shapes need to fit.".format(tag, i)

        # TODO: generate two noisy inputs, see pmlp
        if init == "normal":
            print "[CNN -- {0}, L{1}]: Init shape {2} via Gaussian.".format(tag, i, shape)
            _tmp = {"std": 0.1}
            _tmp = initweight(shape, variant=init, **_tmp) 
        else:
            print "[CNN -- {0}, L{1}]: Init shape {2} via Uniform.".format(tag, i, shape)
            fan_in = np.prod(shape[1:])
            fan_out = (shape[0] * np.prod(shape[2:]) / np.prod(pool))
            winit = np.sqrt(6. / (fan_in + fan_out))
            _tmp = np.asarray(np.random.uniform(low=-winit, high=winit,
                size=shape), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_w{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[0],), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)

        _conv = Tconv.conv2d(input=inpt, filters=_w, filter_shape=shape,
                image_shape=imshape)
        _tmp_name = "{0}_conv_layer{1}".format(tag, i)
        im[_tmp_name] = _conv

        _pool = downsample.max_pool_2d(input=_conv, ds=pool, ignore_border=True)
        _tmp_name = "{0}_pool_layer{1}".format(tag, i)
        im[_tmp_name] = _pool

        inpt = act(_pool + _b.dimshuffle('x', 0, 'x', 'x'))
        _tmp_name = "{0}_cnn_layer{1}".format(tag, i)
        im[_tmp_name] = inpt
        imshape = (imshape[0], shape[0], (imshape[2] - shape[2] + 1)//pool[0],\
                (imshape[3] - shape[3] + 1)//pool[1])
    im[_tmp_name] = im[_tmp_name].flatten(2)
    config['otpt'] = _tmp_name


def deconv(config, params, im):
    """
    Deconvolutional stack. An experimental state.
    """
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']
    pools = config['pools']

    assert len(shapes) == len(activs),\
            "[CNN -- {0}]: One layer, One activation.".format(tag)
    assert len(shapes) == len(pools),\
            "[CNN -- {0}]: One layer, One Pool.".format(tag)

    imshape = config['imshape']

    print "[CNN -- {0}]: CNN with {1} layers, input image {2}.".format(tag, len(shapes), imshape)
    
    init = config["init"]

    inpt = im[config['inpt']]
    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[CNN -- {0}]: Input is a 1-list, taking first element.".format(tag)
            inpt = inpt[0]

    inpt = inpt.reshape(imshape)
    _tmp_name = config['inpt']
    for i, (shape, act, pool) in enumerate(zip(shapes, activs, pools)):
        assert imshape[1] == shape[1],\
            "[CNN -- {0}, L{1}]: Input and Shapes need to fit.".format(tag, i)

        if init == "normal":
            print "[CNN -- {0}, L{1}]: Init shape {2} via Gaussian.".format(tag, i, shape)
            _tmp = {"std": 0.1}
            _tmp = initweight(shape, variant=init, **_tmp) 
        else:
            print "[CNN -- {0}, L{1}]: Init shape {2} via Uniform.".format(tag, i, shape)
            fan_in = np.prod(shape[1:])
            fan_out = (shape[0] * np.prod(shape[2:]) / np.prod(pool))
            winit = np.sqrt(6. / (fan_in + fan_out))
            _tmp = np.asarray(np.random.uniform(low=-winit, high=winit,
                size=shape), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_w{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[0],), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)

        _conv = Tconv.conv2d(input=inpt, filters=_w, filter_shape=shape,
                image_shape=imshape)
        _tmp_name = "{0}_conv_layer{1}".format(tag, i)
        im[_tmp_name] = _conv

        _pool = downsample.max_pool_2d(input=_conv, ds=pool, ignore_border=True)
        _tmp_name = "{0}_pool_layer{1}".format(tag, i)
        im[_tmp_name] = _pool

        inpt = act(_pool + _b.dimshuffle('x', 0, 'x', 'x'))
        _tmp_name = "{0}_cnn_layer{1}".format(tag, i)
        im[_tmp_name] = inpt
        imshape = (imshape[0], shape[0], (imshape[2] - shape[2] + 1)//pool[0],\
                (imshape[3] - shape[3] + 1)//pool[1])
    im[_tmp_name] = im[_tmp_name].flatten(2)
    config['otpt'] = _tmp_name


def sequential(config, params, im):
    """
    A sequential is a meta structure. Connect
    several models together sequentially, e.g. CNN + MLP.
    """
    tag = config['tag']
    components = config['components']
    print "[SEQ -- {0}] Sequential with {1} subcomponents.".format(tag, len(components))

    inpt = config['inpt']
    print "[SEQ -- {0}] Input is {1}.".format(tag, inpt)
    for comp in components:
        assert "type" in comp, "[SEQ -- {0}] Subcomponent needs 'type'.".format(tag)
        typ = comp['type']
 
        _tag = comp['tag']
        comp['tag'] = "|".join([tag, _tag])

        comp['inpt'] = inpt
        typ(config=comp, params=params, im=im)

        assert "otpt" in comp, "[SEQ -- {0}] Subcomponent needs 'otpt'.".format(tag)
        inpt = comp['otpt']

    config['otpt'] = inpt


def parallel(config, params, im):
    """
    A parallel is a meta structure. Connect
    several models together in parallel, e.g. CNN || MLP.
    This is useful for training semi-supervised.
    """
    tag = config['tag']
    components = config['components']
    print "[PAR -- {0}] Parallel with {1} subcomponents.".format(tag, len(components))

    inpt = config['inpt']
    print "[PAR -- {0}] Input is {1}.".format(tag, inpt)
    # TODO handle inpt correctly for parallel types.
    for comp in components:
        assert "type" in comp, "[PAR -- {0}] Subcomponent needs 'type'.".format(tag)
        typ = comp['type']
        
        _tag = comp['tag']
        comp['tag'] = "|".join([tag, _tag])

        comp['inpt'] = inpt
        typ(config=comp, params=params, im=im)
        
        assert "otpt" in comp, "[PAR -- {0}] Subcomponent needs 'otpt'.".format(tag)
    
    config['otpt'] = inpt


def dblin(config, params, im):
    """
    A directed bilinear generative model.
    See , 2011.
    """
    tag = config['tag']

    # different to simple models: pass names of multiple inputs
    # as list of strings, use these strings to access intermediates.
    inpt = config['inpt']

    assert len(inpt) == 2, "[DBLIN -- {0}]: Generative Bilinear Model needs two inputs.".format(tag)

    print "[DBLIN -- {0}]: Generative Bilinear Model.".format(tag)

    if "normalize" in config:
        print "[DBLIN -- {0}]: Normalize weights!".format(tag)
        normalize = config['normalize']
    else:
        normalize = {}

    c = im[inpt[0]]
    d = im[inpt[1]]
    if 'dactiv' in config:
        print "[DBLIN -- {0}]: Preactivation for d: {1}".format(tag, config['dactiv'])
        activ = config['dactiv']
        d = activ(d) 

    # 'theta' == shape of theta matrix
    theta = config['theta']
    _tmp = initweight(theta, variant=config['init']['theta'])
    _tmp_name = "{0}_dblin-theta".format(tag)
    theta = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
    params.append(theta)
    if 'theta' in normalize:
        print "[DBLIN -- {0}]: Normalize theta along axis={1}.".format(tag, normalize['theta'])
        im['normalize'][_tmp_name] = normalize['theta']
 
    c_theta = config['tactiv'](T.dot(c, theta))
    print "[DBLIN -- {0}]: Activation for c_theta: {1}".format(tag, config['tactiv'])
    im['dblin_c_theta'] = c_theta

    # 'psi' in config -> shape of psi matrix
    # normalize rows from psi?
    psi = config['psi']
    _tmp = initweight(psi, variant=config['init']['psi'])
    _tmp_name = "{0}_dblin-psi".format(tag)
    psi = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
    params.append(psi)
    if 'psi' in normalize:
        print "[DBLIN -- {0}]: Normalize psi along axis={1}.".format(tag, normalize['psi'])
        im['normalize'][_tmp_name] = normalize['psi']
 
    d_psi = config['pactiv'](T.dot(d, psi))
    print "[DBLIN -- {0}]: Activation for d_psi: {1}".format(tag, config['pactiv'])
    im['dblin_d_psi'] = d_psi

    a = c_theta * d_psi
    im['dblin_a'] = a

    act = config['activ']
    print "[DBLIN -- {0}]: Activation for a: {1}".format(tag, act)
    a = act(a)
    im['dblin_act(a)'] = a

    # bias for output
    phi = config['phi']
    _tmp = np.zeros((phi[1],), dtype=theano.config.floatX)
    _tmp_name = "{0}_bx".format(tag)
    _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)

    _tmp = initweight(phi, variant=config['init']['phi'])
    _tmp_name = "{0}_dblin-phi".format(tag)
    phi = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
    x = T.dot(a, phi) + _b
    im['dblin_x'] = x
    params.append(phi)
    params.append(_b)
    if 'phi' in normalize:
        print "[DBLIN -- {0}]: Normalize phi along axis={1}.".format(tag, normalize['phi'])
        im['normalize'][_tmp_name] = normalize['phi']
 
    config['otpt'] = 'dblin_x'


def kl_dg_g(config, params, im):
    """
    Kullback-Leibler divergence between diagonal
    gaussian and zero/one gaussian.
    """
    if 'tag' in config:
        tag = config['tag']
    else:
        tag = ''

    inpt = im[config['inpt']]

    dim = inpt.shape[1] / 2
    mu = inpt[:, :dim]
    log_var = inpt[:, dim:]

    _tmp = "{0}_kl_dg_g_mu".format(tag)
    im[_tmp] = mu
    _tmp = "{0}_kl_dg_g_log_var".format(tag)
    im[_tmp] = log_var

    mu_sq = mu * mu
    var = T.exp(log_var)

    rng = T.shared_randomstreams.RandomStreams()
    # gaussian zero/one noise
    gzo = rng.normal(size=mu.shape, dtype=theano.config.floatX)
    # Reparameterized latent variable
    z = mu + T.sqrt(var+1e-4)*gzo
    _tmp = "{0}_kl_dg_g_z".format(tag)
    im[_tmp] = z
    config['otpt'] = _tmp

    # difference to paper: gradient _descent_, minimize an upper bound
    # -> needs a negative sign
    cost = -(1 + log_var - mu_sq - var)
    cost = T.sum(cost, axis=1)
    cost = 0.5 * T.mean(cost)
    _tmp = "{0}_kl_dg_g".format(tag)
    im[_tmp] = cost
    im['cost'] = im['cost'] + cost


def kl_lrg_g(config, params, im):
    """
    Kullback-Leibler divergence between low-rank
    gaussian and zero/one gaussian.

    From Stochastic Back-propagation and Variational Inference in
    Deep Latent Gaussian Models, eq. 16, eq. 17. Note that in v2
    of the arxiv pdf, eq. 16 has a tiny sign mistake, which shows
    in the trace computation.
    """
    if 'tag' in config:
        tag = config['tag']
    else:
        tag = ''

    inpt = im[config['inpt']]

    dim = inpt.shape[1] / 3
    mu = inpt[:, :dim]
    
    # Note: for convenience, this is the inverse of the log_var
    # See eq. 16, the inverse is always used in the computations.
    log_var_inv = inpt[:, dim:2*dim]
    # avoid overflow?
    #var_inv = inpt[:, dim:2*dim]**2

    # 1-d direction of orientation
    u = inpt[:, 2*dim:]

    _tmp = "{0}_kl_lrg_g_mu".format(tag)
    im[_tmp] = mu
    _tmp = "{0}_kl_lrg_g_log_var_inv".format(tag)
    im[_tmp] = log_var_inv
    #_tmp = "{0}_kl_lrg_g_var_inv".format(tag)
    #im[_tmp] = var_inv

    _tmp = "{0}_kl_lrg_g_u".format(tag)
    im[_tmp] = u
 
    mu_sq = mu * mu
    var_inv = T.exp(log_var_inv)
    #log_var_inv = T.log(var_inv+1e-8)

    # get log determinant
    # Du is D-1 * u in the paper
    Du = var_inv * u
    uDu = T.sum(u*Du, axis=1).dimshuffle(0, 'x')
    eta = 1./(uDu + 1)
    _tmp = "{0}_kl_lrg_g_eta".format(tag)
    im[_tmp] = eta
    _tmp = "{0}_kl_lrg_g_logeta".format(tag)
    im[_tmp] = T.log(eta+1e-8)
    _tmp = "{0}_kl_lrg_g_detD".format(tag)
    im[_tmp] = T.sum(log_var_inv, axis=1).dimshuffle(0, 'x')
    logDet = T.log(eta) + T.sum(log_var_inv, axis=1).dimshuffle(0, 'x')
    _tmp = "{0}_kl_lrg_g_logDet".format(tag)
    im[_tmp] = logDet

    # get trace (use some previous computations)
    Dusq = Du * Du
    # the minus here is newish
    trc =  T.sum(var_inv, axis=1).dimshuffle(0, 'x') - eta*T.sum(Dusq, axis=1).dimshuffle(0, 'x')
    _tmp = "{0}_kl_lrg_g_trc".format(tag)
    im[_tmp] = trc

    # generate samples
    rng = T.shared_randomstreams.RandomStreams()
    # gaussian zero/one noise
    gzo = rng.normal(size=mu.shape, dtype=theano.config.floatX)
    trf = T.sum(u*T.sqrt(var_inv+1e-8)*gzo, axis=1).dimshuffle(0, 'x')
    gzo = T.sqrt(var_inv+1e-8)*gzo - (1-T.sqrt(eta))/(uDu+1e-6) * trf * Du
    # Reparameterized latent variable
    z = mu + gzo
    _tmp = "{0}_lrg_dg_g_z".format(tag)
    im[_tmp] = z
    config['otpt'] = _tmp

    # difference to paper: gradient _descent_, minimize an upper bound
    # -> needs a negative sign
    cost = T.sum(mu_sq-1, axis=1) + trc - logDet
    cost = 0.5 * T.mean(cost)
    _tmp = "{0}_kl_dg_g".format(tag)
    im[_tmp] = cost
    im['cost'] = im['cost'] + cost


def kl_dlap_lap(config, params, im):
    """
    Kullback-Leibler divergnence between diagonal
    laplacian and zero/one laplacian.
    """
    if 'tag' in config:
        tag = config['tag']
    else:
        tag = ''

    inpt = im[config['inpt']]

    dim = inpt.shape[1] / 2
    mu = inpt[:, :dim]
    ln_b = inpt[:, dim:]

    _tmp = "{0}_kl_dlap_lap_mu".format(tag)
    im[_tmp] = mu
    _tmp = "{0}_kl_dlap_lap_log_b".format(tag)
    im[_tmp] = ln_b

    mu_sq = mu * mu
    b = T.exp(ln_b)

    rng = T.shared_randomstreams.RandomStreams()
    # uniform -1/2;1/2
    uni = rng.uniform(size=mu.shape, low=-0.5, high=0.5, dtype=theano.config.floatX)
    # Reparameterized latent variable, see e.g. Wikipedia
    z = mu - b*T.sgn(uni)*T.log(1 - 2*T.abs_(uni) + 1e-6)
    _tmp = "{0}_kl_dlap_lap_z".format(tag)
    im[_tmp] = z
    config['otpt'] = _tmp 

    # difference to paper: gradient _descent_, minimize an upper bound
    # -> needs a negative sign
    cost = -ln_b + b*T.exp(-T.abs_(mu)/b) + T.abs_(mu) - 1
    cost = T.sum(cost, axis=1)
    cost = T.mean(cost)
    _tmp = "{0}_kl_dlap_lap".format(tag)
    im[_tmp] = cost
    im['cost'] = im['cost'] + cost


def rim(config, params, im):
    """
    """
    if 'tag' in config:
        tag = config['tag']
    else:
        tag = ''

    inpt = im[config['inpt']]

    # conditional entropy
    cond_entropy = -inpt * T.log(inpt).sum(axis=1)
    cond_entropy = cond_entropy.mean()

    # marginal entropy
    marginal = inpt.mean(axis=0)
    entropy = -marginal * T.log(marginal).sum(axis=1)
    entropy = entropy.mean()

    # minimize negative mutual information
    cost = cond_entropy - entropy
    im['cost'] = im['cost'] + cost


def multi_kl(config, params, im):
    """
    multiple kl divergences, parallel.
    """
    kls = config['kls'] # the list of kl divergences.
    print "[MULTIKL]: Combined {0} KL divergences.".format(len(kls))
    print "[MULTIKL]: Note: Cost handling is done internally!"


    inpt = im[config['inpt']]
    idx = 0
    otpt = []
    #
    for j, kl in enumerate(kls):
        if 'tag' in kl:
            kl['tag'] = "multikl-{0}-{1}".format(j, kl['tag'])
        else:
            kl['tag'] = "multikl-{0}".format(j)
        typ = kl['type']
        units = kl['units']
        suff = kl['suff']
        _tmp = "{0}_inpt_{1}".format(j, kl['type'])
        kl['inpt'] = _tmp
        im[_tmp] = inpt[:, idx:idx+units*suff]
        typ(kl, params, im)
        otpt.append(kl['otpt'])
        idx = idx + units*suff
    #
    print "[MULTIKL]: Total size of representation spread over {0} parts: {1}.".format(len(kls), idx)
    config['otpt'] = tuple(otpt)


def bern_xe(config, params, im):
    """
    Bernoulli cross entropy.

    Used for predicting binary
    variables, needs a target.
    """
    inpt = im[config['inpt']]
    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[BERNXE]: Input is a 1-list, taking first element."
            inpt = inpt[0]
    
    t = im[config['trgt']]
    if type(t) in [list, tuple]:
        if len(t) == 1:
            print "[BERNXE]: Target is a 1-list, taking first element."
            t = t[0]
    
    pred = T.nnet.sigmoid(inpt)
    im['predict'] = pred 
    # difference to paper: gradient _descent_, minimize upper bound
    # -> needs a negative sign
    #cost= -T.nnet.binary_crossentropy(pred, t)
    cost = -(t*T.log(pred + 1e-4) + (1-t)*T.log(1-pred + 1e-4))
    im['neg_log_like_per_sampel'] = cost
    cost = T.sum(cost, axis=1)
    cost = T.mean(cost)
    im['bern_xe'] = cost
    im['cost'] = im['cost'] + cost


def g_nll(config, params, im):
    """
    Gaussian likelihood, fixed variance for all output dimensions.

    Used for predicting real valued
    variables, needs a target.
    """
    inpt = im[config['inpt']]
    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[G_NLL]: Input is a 1-list, taking first element."
            inpt = inpt[0]
    
    t = im[config['trgt']]
    if type(t) in [list, tuple]:
        if len(t) == 1:
            print "[G_NLL]: Target is a 1-list, taking first element."
            t = t[0]

    scale = config['scale']
    if 'sigma' in config:
        sigma = config['sigma']
    else:
        sigma = 1
    pred = scale*inpt
    im['predict'] = pred 
    # difference to paper: gradient _descent_, minimize upper bound
    # -> needs a negative sign
    cost = ((pred - t)/sigma 	)**2
    cost = T.sum(cost, axis=1)
    cost = 0.5 * T.mean(cost)
    im['g_nll'] = cost
    im['cost'] = im['cost'] + cost


def dg_nll(config, params, im):
    """
    Diagonal Gaussian likelihood -- mean and variance.

    Used for predicting real valued
    variables with individual variance, needs a target.
    """
    inpt = im[config['inpt']]

    dim = inpt.shape[1] / 2
    mu = inpt[:, :dim]
    log_var = inpt[:, dim:]
    var = T.exp(log_var)

    if type(inpt) in [list, tuple]:
        if len(inpt) == 1:
            print "[G_NLL]: Input is a 1-list, taking first element."
            inpt = inpt[0]
    
    t = im[config['trgt']]
    if type(t) in [list, tuple]:
        if len(t) == 1:
            print "[G_NLL]: Target is a 1-list, taking first element."
            t = t[0]

    scale = config['scale']
    pred = scale*mu
    im['predict'] = pred
    im['log_var'] = log_var
    # difference to paper: gradient _descent_, minimize upper bound
    # -> needs a negative sign
    cost = ((pred - t)**2)/var 
    cost = 0.5*T.sum(cost, axis=1) + dim/2.*T.log(2*np.pi) + T.sum(log_var, axis=1) 
    cost = T.mean(cost)
    im['dg_nll'] = cost
    im['cost'] = im['cost'] + cost



# example usage with digaonal gauss KL ('ims' are intermediates from vae)
# params = [ims['_kl_dg_g_mu'], ims['_kl_dg_g_log_var']]
# res = estimate_ll(data[:1000], 500, ims['inpt'][0], ims['_kl_dg_g_z'], ims['neg_log_like_per_sampel'], params, ll_diag_gauss_logvar, [0, 0])
# Note: pz is ll_diag_gauss_logvar, using log variance -> params are 0 and _0_!
def estimate_ll(data, no_mcs, inpts, z_samples, ncll_x_z, params, 
        pz, prior):
    """Estimate loglikelihood of samples in _data_, using _no_mcs_
    many importance samples from proposal distribution _pz_. _pz_
    has parameters -- for the posterior approximation these are in
    _params_ (theano expression per parameter), for the prior these
    are in _prior_ (e.g. simply [0, 1] for a standard normal).
    _inpts_ is a theano expression for the input of the VAE,
    _z_samples_ is a theano expression for generating samples from
    the proposal distribution given _inpts_ (_z_samples_ and _pz_ 
    must be matching), _ncll_x_z_ is a theano expression of the
    negative loglikelihood of x given z (z representing the latents
    of the graphical model).
    """
    z_smpls = theano.function([inpts], z_samples, allow_input_downcast=True)
    params = theano.function([inpts], params, allow_input_downcast=True)
    ncll_x_z = theano.function([inpts, z_samples], ncll_x_z, allow_input_downcast=True)
    mc_ll = np.zeros((no_mcs, data.shape[0]))
    for i in xrange(no_mcs):
        zs = z_smpls(data)
        pars = params(data)
        ll_z_x = pz(zs, *pars).sum(axis=1)
        ll_z = pz(zs, *prior).sum(axis=1)
        ll_x_z = -ncll_x_z(data, zs).sum(axis=1)
        ll_xz = ll_x_z + ll_z - ll_z_x
        mc_ll[i, :] = ll_xz
    ll = logsumexp(mc_ll, axis=0) - np.log(no_mcs)
    return ll.mean()


# estimating loglikelihood per row of samples x from diagonal gauss, with
# parameters mu (mean) and logvar (variance in log). 
# Needs to sum of axis=1 to get the estimate for one sample (a row) in x.
def ll_diag_gauss_logvar(x, mu, logvar):
    const = -np.log(2*np.pi)/2
    return const -logvar/2 - (x - mu)**2 / (2 * np.exp(logvar))


def vae(config, special=None, tied=None):
    """
    Variational Autoencoder. Provide information on
    _encoder_ and _decoder_ in these dictionaries.
    _tied_ indicates wether the first layer of
    the encoder and the last layer of the decoder are tied.

    See:
    Autoencoding Variational Bayes by Kingma, Welling. 2014.
    """
    # journey starts here:
    x = T.matrix('inpt')

    # collect intermediate expressions
    intermediates = {'inpt': (x,), 'normalize': {}}
    encoder = config['encoder']
    decoder = config['decoder']
    kl_cost = config['kl']
    g_cost = config['cost']

    # collect parameters
    params = []

    # collect normalizations

    # cost
    intermediates['cost'] = 0

    enc = encoder['type']
    # encoder needs a field for input -- name of 
    # intermediate symbolic expr.
    encoder['inpt'] = 'inpt'
    enc(config=encoder, params=params, im=intermediates)
    assert "otpt" in encoder, "Encoder needs an output."
    
    kl = kl_cost['type']
    kl_cost['inpt'] = encoder['otpt']
    kl(config=kl_cost, params=params, im=intermediates)
    assert "otpt" in kl_cost, "KL_cost needs to sample an output."

    dec = decoder['type']
    decoder['inpt'] = kl_cost['otpt']
    dec(config=decoder, params=params, im=intermediates)
    assert "otpt" in decoder, "Decoder needs an output."

    cost = g_cost['type']
    g_cost['inpt'] = decoder['otpt']
    g_cost['trgt'] = 'inpt'
    cost(config=g_cost, params=params, im=intermediates)

    cost = intermediates['cost']
    return cost, params, intermediates


def semi_vae(config, special=None, tied=None):
    """
    """
    # journey starts here:
    x = T.matrix('inpt')

    # collect intermediate expressions
    intermediates = {'inpt': (x,), 'normalize': {}}
    encoder = config['encoder']
    decoder = config['decoder']
    kl_cost = config['kl']
    g_cost = config['cost']

    # collect parameters
    params = []

    # collect normalizations

    # cost
    intermediates['cost'] = 0

    enc = encoder['type']
    # encoder needs a field for input -- name of 
    # intermediate symbolic expr.
    encoder['inpt'] = 'inpt'
    enc(config=encoder, params=params, im=intermediates)
    assert "otpt" in encoder, "Encoder needs an output."
    
    kl = kl_cost['type']
    kl_cost['inpt'] = encoder['otpt']
    kl(config=kl_cost, params=params, im=intermediates)
    assert "otpt" in kl_cost, "KL_cost needs to sample an output."

    dec = decoder['type']
    decoder['inpt'] = kl_cost['otpt']
    dec(config=decoder, params=params, im=intermediates)
    assert "otpt" in decoder, "Decoder needs an output."

    cost = g_cost['type']
    g_cost['inpt'] = decoder['otpt']
    g_cost['trgt'] = 'inpt'
    cost(config=g_cost, params=params, im=intermediates)

    cost = intermediates['cost']
    return cost, params, intermediates


def adadelta(params, grads, settings, **kwargs):
    """
    AdaDELTA, by Matthew Zeiler.
    """
    eps = 10e-8
    lr = settings['lr']
    decay = settings['decay']
    print "[AdaDELTA] lr: {0}; decay: {1}".format(lr, decay)

    # Average Root Mean Squared (arms) Gradients
    arms_grads = []
    # Average Root Mean Squared (arms) Updates
    arms_upds = []
    for p in params:
        arms_grad_i = theano.shared(np.zeros(p.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        arms_grads.append(arms_grad_i)
        #
        arms_upds_i = theano.shared(np.zeros(p.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        arms_upds.append(arms_upds_i)

    updates = OrderedDict()
    for grad_i, arms_grad_i in zip(grads, arms_grads):
        updates[arms_grad_i] = decay*arms_grad_i + (1-decay)*grad_i*grad_i

    for param_i, grad_i, arms_grad_i, arms_upds_i in zip(params, grads, arms_grads, arms_upds):
        delta_i = T.sqrt(arms_upds_i + eps)/T.sqrt(updates[arms_grad_i] + eps) * lr * grad_i
        updates[arms_upds_i] = decay*arms_upds_i + (1-decay)*delta_i*delta_i
        
        if param_i.name in kwargs:
            up = param_i - delta_i
            wl = T.sqrt(T.sum(T.square(up), axis=kwargs[param_i.name]) + eps)
            if kwargs[param_i.name] == 0:
                updates[param_i] = up/wl
            else:
                updates[param_i] = up/wl.dimshuffle(0, 'x')
            print "[AdaDELTA] Normalized {0} along axis {1}, EPS={2}".format(param_i.name, kwargs[param_i.name], eps)
        else:
            updates[param_i] = param_i - delta_i
    return updates


def adam(params, grads, settings, **kwargs):
    """
    Adam by Kingma and Ba.

    TODO
    """
    return updates


def rmsprop(params, grads, settings, **kwargs):
    """
    RMSprop.
    """
    eps = 10e-8
    lr = settings['lr']
    decay = settings['decay']
    mom = settings['momentum']

    print "[RMSprop] lr: {0}; decay: {1}, momentum: {2}".format(lr, decay, mom)

    # Average Root Mean Squared (arms) Gradients
    arms_grads = []
    # Average Root Mean Squared (arms) Updates
    _momentums = []
    for p in params:
        arms_grad_i = theano.shared(np.zeros(p.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        arms_grads.append(arms_grad_i)
        #
        p_mom = theano.shared(np.zeros(p.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        _momentums.append(p_mom)

    updates = OrderedDict()
    for grad_i, arms_grad_i in zip(grads, arms_grads):
        updates[arms_grad_i] = decay*arms_grad_i + (1-decay)*grad_i*grad_i

    for param_i, grad_i, arms_grad_i, mom_i in zip(params, grads, arms_grads, _momentums):
        delta_i = lr*grad_i/T.sqrt(updates[arms_grad_i] + eps)
        updates[mom_i] = mom*mom_i + delta_i
        
        if param_i.name in kwargs:
            up = param_i - delta_i
            wl = T.sqrt(T.sum(T.square(up), axis=kwargs[param_i.name]) + 1e-8)
            if kwargs[param_i.name] == 0:
                updates[param_i] = up/wl
            else:
                updates[param_i] = up/wl.dimshuffle(0, 'x')
            print "[RMSprop] Normalized {0} along axis {1}".format(param_i.name, kwargs[param_i.name])
        else:
            updates[param_i] = param_i - delta_i
    return updates


def idty(x):
    return x


def relu(x):
    return T.maximum(x, 0)


def softplus(x):
    return T.log(1 + T.exp(x))


def test_vae(enc_out=2, epochs=100, lr=1,
        momentum=0.9, decay=0.9, btsz=100):
    """
    Test variational autoencdoer on MNIST.
    This needs mnist.pkl.gz in your directory.
    AdaDelta seems to perform better.
    """
    import gzip, cPickle
    mnist_f = gzip.open("mnist.pkl.gz",'rb')
    train_set, valid_set, test_set = cPickle.load(mnist_f)
    data = train_set[0]
    mnist_f.close()

    batches = data.shape[0]/btsz
    print "Variational AE"
    print "Epochs", epochs
    print "Batches per epoch", batches
    print "lr:{0}, momentum:{1}".format(lr, momentum)
    print

    # specify encoder
    enc = {
        'tag': 'enc',
        'type': mlp,
        'shapes': [(28*28, 200), (200, enc_out*2)],
        'activs': [T.tanh, idty],
        'init': "normal",
        'cost': {
            'type': kl_dg_g,
        }
    }

    # 'inpt': 'z', this should be automatic by lookup of encoder type.
    dec = {
        'tag': 'dec',
        'type': mlp,
        'inpt': 'z',
        'shapes': [(enc_out, 200), (200, 28*28)],
        'activs': [T.tanh, idty],
        'init': "normal",
        'cost': {
            'type': bern_xe,
        }
    }

    cost, params, ims = vae(encoder=enc, decoder=dec)
    grads = T.grad(cost, params)

    learner = {"lr": lr, "momentum": momentum, "decay": decay}
    #updates = momntm(params, grads, **learner)
    updates = adadelta(params, grads, **learner)
    train = theano.function([ims['inpt']], cost, updates=updates, allow_input_downcast=True)

    sz = data.shape[0]
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            cost += btsz*train(data[mbi*btsz:(mbi+1)*btsz])
        print epoch, cost/sz
