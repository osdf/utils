"""
Some theano offspring.
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


def initweight(shape, variant="normal", **kwargs):
    """
    Init weights.
    """
    if variant is "normal":
        if "std" in kwargs:
            std = kwargs["std"]
        else:
            std = 0.1
        weights = np.asarray(std * np.random.standard_normal(shape), dtype=theano.config.floatX)
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


def mlp(config, params, im):
    """
    A mlp acting as an encoding or decoding layer --
    This depends on the loss that is specified
    in the _config_ dictionary.
    """
    shapes = config['shapes']
    activs = config['activs']

    inpt = im[config['inpt']]

    _tmp_name = config['inpt']
    for i, (shape, act) in enumerate(zip(shapes, activs)):
        # fully connected
        _tmp = initweight(layer[0], variant=config["init"])
        _tmp_name = "enc_w{0}".format(i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        # bias
        _tmp = np.zeros((layer[0][1],), dtype=theano.config.floatX)
        _tmp_name = "enc_b{0}".format(i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)
        inpt = act(T.dot(inpt, _w) + _b)
        _tmp_name = "enc_layer{0}".format(i)
        im[_tmp_name] = inpt

    print "[MLP] building cost."
    print "[MLP] designated input:", _tmp_name
    cost_conf = config['cost']
    cost_conf['inpt'] = _tmp_name

    cost = cost_conf['type']
    loss = cost(config=cost_conf, params=params, im=im)


def kl_dg_g(config, params, im):
    """
    Kullback-Leibler divergnence between diagonal
    gaussian and zero/one gaussian.
    """
    inpt = im[config['inpt']]
    dim = config['dim']

    mu = inpt[:, :dim]
    log_var = inpt[:, dim:]

    im['kl_dg_g_mu'] = mu
    im['kl_dg_g_log_var'] = log_var

    mu_sq = mu * mu
    var = T.exp(log_var)

    rng = T.shared_randomstreams.RandomStreams()
    # gaussian zero/one noise
    gzo = rng.normal(size=mu.shape)
    # Reparameterized latent variable
    z = mu + T.sqrt(var+1e-4)*gzo
    im['z'] = z
    
    # difference to paper: gradient _descent_, minimize an upper bound
    # -> needs a negative sign
    cost = -(1 + log_var - mu_sq - var)
    cost = T.sum(cost, axis=1)
    cost = 0.5 * T.mean(cost)
    return cost, z, params


def decoder_bern(z, t, W1init, W2init, activ=T.tanh):
    """
    A decoder representing a bernoulli vector. Generate the observations
    from _z_, compare to targets _t_.
    """
    W1 = theano.shared(np.asarray(W1init, dtype=theano.config.floatX),
        borrow=True, name='W1_dec_bern')
    _b1 = np.zeros((W1init.shape[1],), dtype=theano.config.floatX)
    b1 = theano.shared(value=_b1, borrow=True, name="b1_dec")

    h = activ(T.dot(z, W1) + b1)

    W2 = theano.shared(np.asarray(W2init, dtype=theano.config.floatX),
        borrow=True, name='W2_dec_bern')
    _b2 = np.zeros((W2init.shape[1],), dtype=theano.config.floatX)
    b2 = theano.shared(value=_b2, borrow=True, name="b2_dec")

    bern = T.nnet.sigmoid(T.dot(h, W2) + b2)

    # difference to paper: gradient _descent_, minimize upper bound
    # -> needs a negative sign
    cost = -(t*T.log(bern + 1e-4) + (1-t)*T.log(1-bern + 1e-4))
    cost = T.sum(cost, axis=1)
    cost = T.mean(cost)
    params = [W1, b1, W2, b2]
    return cost, bern, params


def vae(encoder, decoder, platent, tied=False):
    """
    Variational Autoencoder. Provide information on
    _encoder_ and _decoder_ in these dictionaries.
    _platent_ represents the reparametrized latent variable
    layer. _tied_ indicates wether the first layer of
    the encoder and the last layer of the decoder are tied.

    See:
    Autoencoding Variational Bayes by Kingma, Welling. 2014.
    """
    # journey starts here:
    x = T.matrix('x')

    # collect intermediate expressions, just in case.
    intermediates = {'inpt': x}

    # collect parameters
    params = []

    # cost
    cost = 0

    enc = encoder['type']
    pass_over = enc(config=encoder, params=params, im=intermediates)

    plat = platent['type']
    cost_latent = plat(config=platent, params=params, im=intermediates)
    cost = cost + cost_latent

    dec = decoder['type']
    cost_dec = dec(config=decoder, params=params, im=intermediates, passed=pass_over)
    cost = cost + cost_dec

    return intermediates, cost, params


def adadelta(params, grads, **kwargs):
    """
    AdaDELTA, by Matthew Zeiler.
    """
    eps = 10e-8
    print "OPTIMIZER: AdaDELTA"
    lr = kwargs['lr']
    decay = kwargs['decay']

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
            wl = T.sqrt(T.sum(T.square(up), axis=kwargs[param_i.name]) + 1e-8)
            if kwargs[param_i.name] == 0:
                updates[param_i] = up/wl
            else:
                updates[param_i] = up/wl.dimshuffle(0, 'x')
            print "Normalized {0} along axis {1}".format(param_i.name, kwargs[param_i.name])
        else:
            updates[param_i] = param_i - delta_i
    return updates


def test_vae(enc_hidden=500, enc_out=2, dec_hidden=500,
        epochs=100, lr=0.0001, momentum=0.9, btsz=100):
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

    # specify decoder
    mlp_enc = [(data.shape[1], enc_hidden), (enc_hidden, enc_out),
            (enc_hidden, enc_out)]
    encoder = encoder_dg

    # specify encoder
    mlp_dec = [(enc_out, dec_hidden), (dec_hidden, data.shape[1])]
    decoder = decoder_bern

    params, cost, grads, x, z, rec = vae(encoder, decoder, mlp_enc, mlp_dec)
    learner = {"lr": lr, "momentum": momentum}
    updates = momntm(params, grads, **learner)
    train = theano.function([x], cost, updates=updates, allow_input_downcast=True)
    # get data
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            cost += btsz*train(data[mbi*btsz:(mbi+1)*btsz])
        print epoch, cost/data.shape[0]
    return params
