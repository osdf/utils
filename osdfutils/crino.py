"""
Some theano offspring.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


# list of cost intermediates
vae_cost_ims = {}
vae_handover = {}


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
        weights = np.asarray(np.random.normal(loc=0, scale=std, size=shape),
                dtype=theano.config.floatX)
    elif variant is "uniform":
        fan_in = kwargs["fan_in"]
        fan_out = kwargs["fan_out"] 
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
    print "lr: {0}; momentum: {1}".format(lr, momentum)

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
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']

    assert len(shapes) == len(activs),\
            "[MLP -- {0}]: One layer, One activation.".format(tag)

    if 'tied' in config:
        tied = config['tied']
    else:
        tied = {}

    inpt = im[config['inpt']][0]

    _tmp_name = config['inpt']
    for i, (shape, act) in enumerate(zip(shapes, activs)):
        # fully connected
        if i in tied:
            _tied = False
            for p in params:
                if tied[i] == p.name:
                    print "Tieing layer {0} in {1} with {2}".format(i, tag, p.name)
                    _w = p.T
                    _w = _w[:shape[0], :shape[1]]
                    _tied = True
            assert _tied,\
                    "[MLP -- {0}]: Tieing was set for layer {1}, but unfulfilled!".format(tag, i)
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

    print "[MLP -- {0}] building cost.".format(tag)
    print "[MLP -- {0}] its designated input: {1}".format(tag, _tmp_name)
    cost_conf = config['cost']
    cost_conf['inpt'] = _tmp_name

    cost = cost_conf['type']
    loss = cost(config=cost_conf, params=params, im=im)
    return loss


def pmlp(config, params, im):
    """
    A pmlp acting as an encoding or decoding layer --
    This depends on the loss that is specified
    in the _config_ dictionary. A 'pmlp' is a
    MLP with product interactions
    """
    tag = config['tag']

    shapes = config['shapes']
    activs = config['activs']

    assert len(shapes) == len(activs),\
            "[MLP -- {0}]: One layer, One activation.".format(tag)

    inpt = im[config['inpt']]

    _tmp_name = config['inpt']
    for i, (shape, act) in enumerate(zip(shapes, activs)):
        # fully connected
        _tmp = initweight(shape, variant=config["init"])
        _tmp_name = "{0}_w{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[1],), dtype=theano.config.floatX)
        _tmp_name = "{0}_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)
        inpt = act(T.dot(inpt, _w) + _b)
        _tmp_name = "{0}_layer{1}".format(tag, i)
        im[_tmp_name] = inpt

    print "[MLP -- {0}] building cost.".format(tag)
    print "[MLP -- {0}] its designated input: {1}".format(tag, _tmp_name)
    cost_conf = config['cost']
    cost_conf['inpt'] = _tmp_name

    cost = cost_conf['type']
    loss = cost(config=cost_conf, params=params, im=im)
    return loss


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

    init = config["init"]

    inpt = im[config['inpt']][0]
    inpt = inpt.reshape(imshape)
    _tmp_name = config['inpt']
    for i, (shape, act, pool) in enumerate(zip(shapes, activs, pools)):
        assert imshape[1] == shape[1],\
            "[CNN -- {0}, L{1}]: Input and Shapes need to fit.".format(tag, i)
    
        if init == "normal":
            print "[CNN -- {0}]: Init via Gaussian.".format(tag)
            _tmp = initweight(shape, variant=init, {"std": 0.1}) 
        else:
            print "[CNN -- {0}]: Init via Uniform.".format(tag)
            fan_in = np.prod(shape[1:])
            fan_out = (shape[0] * np.prod(shape[2:]) / np.prod(pool))
            winit = np.sqrt(6. / (fan_in + fan_out))
            _tmp = np.asarray(np.random.uniform(low=-winit, high=winit,
                size=shape), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_w{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[1],), dtype=theano.config.floatX)
        _tmp_name = "{0}_cnn_b{1}".format(tag, i)
        _b = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_b)

        _conv = conv.conv2d(input=inpt, filters=_w, filter_shape=shape,
                image_shape=imshape)
        _tmp_name = "{0}_conv_layer{1}".format(tag, i)
        im[_tmp_name] = _conv
        
        _pool = downsample.max_pool_2d(input=_conv, ds=pool, ignore_border=True)
        _tmp_name = "{0}_pool_layer{1}".format(tag, i)
        im[_tmp_name] = _pool
        
        inpt = act(_pool + _b.dimshuffle('x', 0, 'x', 'x'))
        _tmp_name = "{0}_cnn_layer{1}".format(tag, i)
        im[_tmp_name] = inpt
        imshape = (imshape[0], shape[1], (imshape[0] - shape[2] + 1)//pool[0],\
                (imshape[1] - shape[3] + 1)//pool[1])

    im[_tmp_name] = im[_tmp_name].flatten(2)
    config['handover'] = (_tmp_name,)
    
    return loss


def composite(config, params, im):
    """
    A composite is a meta structure. Connect
    several models together, e.g. CNN + MLP.
    """
    tag = config['tag']
    components = config['components']

    inpt = im[config['inpt']]
    
    for comp in components:


    print "[COMP -- {0}] building cost.".format(tag)
    print "[COMP -- {0}] its designated input: {1}".format(tag, _tmp_name)
    cost_conf = config['cost']
    cost_conf['inpt'] = _tmp_name

    cost = cost_conf['type']
    loss = cost(config=cost_conf, params=params, im=im)
    return loss


def kl_dg_g(config, params, im):
    """
    Kullback-Leibler divergnence between diagonal
    gaussian and zero/one gaussian.
    """
    inpt = im[config['inpt']]

    dim = inpt.shape[1] / 2
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
    im['kl_dg_g'] = cost
    return cost


vae_cost_ims[kl_dg_g] = ('kl_dg_g_mu', 'kl_dg_g_log_var', 'kl_dg_g')
vae_handover[kl_dg_g] = ('z',)


def kl_dlap_lap(config, params, im):
    """
    Kullback-Leibler divergnence between diagonal
    laplacian and zero/one laplacian.
    """
    inpt = im[config['inpt']]

    dim = inpt.shape[1] / 2
    mu = inpt[:, :dim]
    ln_b = inpt[:, dim:]

    im['kl_dlap_lap_mu'] = mu
    im['kl_dlap_lap_log_b'] = ln_b

    mu_sq = mu * mu
    b = T.exp(ln_b)

    rng = T.shared_randomstreams.RandomStreams()
    # uniform -1/2;1/2
    uni = rng.uniform(size=mu.shape, low=-0.5, high=0.5)
    # Reparameterized latent variable, see e.g. Wikipedia
    z = mu - b*T.sgn(uni)*T.log(1 - 2*T.abs_(uni))
    im['z'] = z
    
    # difference to paper: gradient _descent_, minimize an upper bound
    # -> needs a negative sign
    cost = -ln_b + b*T.exp(-T.abs_(mu)/b) + T.abs_(mu) - 1
    cost = T.sum(cost, axis=1)
    cost = T.mean(cost)
    im['kl_dlap_lap'] = cost
    return cost


vae_cost_ims[kl_dlap_lap] = ('kl_dlap_lap_mu', 'kl_dlap_laplog_var', 'kl_dlap_lap')
vae_handover[kl_dlap_lap] = ('z', )


def bern_xe(config, params, im):
    """
    Bernoulli cross entropy.

    Used for predicting binary
    variables, needs a target.
    """
    inpt = im[config['inpt']]
    t = im[config['trgt']]
    
    pred = T.nnet.sigmoid(inpt)
    im['predict'] = pred 
    # difference to paper: gradient _descent_, minimize upper bound
    # -> needs a negative sign
    cost = -(t*T.log(pred + 1e-4) + (1-t)*T.log(1-pred + 1e-4))
    cost = T.sum(cost, axis=1)
    cost = T.mean(cost)
    im['bern_xe'] = cost
    return cost


vae_cost_ims[bern_xe] = ('predict', 'bern_xe')


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

    # collect intermediate expressions, just in case.
    intermediates = {'enc_inpt': (x,)}

    encoder = config['encoder']
    decoder = config['decoder']
    kl_cost = config['kl']
    g_cost = config['cost']

    # collect parameters
    params = []

    # cost
    intermediates['cost'] = 0

    enc = encoder['type']
    # encoder needs a field for input -- name of 
    # intermediate symbolic expr.
    encoder['inpt'] = 'enc_inpt'
    enc(config=encoder, params=params, im=intermediates)
    assert "otpt" in encoder, "Encoder needs an output."

    kl = kl_cost['type']
    kl_cost['inpt'] = encoder['otpt']
    kl(config=kl, params=params, im=intermediates)
    assert "otpt" in kl_cost, "KL_cost needs to sample an output."

    dec = decoder['type']
    decoder['inpt'] = kl_cost['otpt'] 
    dec(config=decoder, params=params, im=intermediates)
    assert "otpt" in kl_cost, "Decoder needs an output."

    cost = g_cost['type']
    g_cost['inpt'] = decoder['otpt']
    cost(config=g_cost, params=params, im=intermediates)

    return cost, params, intermediates


def adadelta(params, grads, **kwargs):
    """
    AdaDELTA, by Matthew Zeiler.
    """
    eps = 10e-8
    print "OPTIMIZER: AdaDELTA"
    lr = kwargs['lr']
    decay = kwargs['decay']
    print "lr: {0}; decay: {1}".format(lr, decay)

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


def rmsprop(params, grads, **kwargs):
    """
    RMSprop.
    """
    eps = 10e-8
    print "OPTIMIZER: rmsPROP"
    lr = kwargs['lr']
    decay = kwargs['decay']
    mom = kwargs['momentum']

    print "lr: {0}; decay: {1}, momentum: {2}".format(lr, decay, momentum)

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
            print "Normalized {0} along axis {1}".format(param_i.name, kwargs[param_i.name])
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
