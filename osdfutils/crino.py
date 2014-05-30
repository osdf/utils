"""
Some theano offspring.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv as Tconv


# list of cost intermediates
vae_cost_ims = {}


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
    lr = kwargs['lr']
    momentum = kwargs['momentum']
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
    #should w1 and w2 be shared?
    _tmp_name = config['inpt']
    for i, (shape, act, noise) in enumerate(zip(shapes, activs, noises)):
        
        if noise[0] == "01":
            inpt1 = rng.binomial(size=inpt.shape, n=1, p=1.0-noise[1], dtype=theano.config.floatX) * inpt
            inpt2 = rng.binomial(size=inpt.shape, n=1, p=1.0-noise[1], dtype=theano.config.floatX) * inpt
        elif noise[0] == "gauss":
            inpt1 = rng.normal(size=inpt.shape, std=noise[1], dtype=theano.config.floatX) + intp
            inpt2 = rng.normal(size=inpt.shape, std=noise[1], dtype=theano.config.floatX) + intp
        else:
            assert False, "[PMLP -- {0}: Unknown noise process.".format(tag)
        
        _tmp = initweight(shape, variant=config["init"])
        _tmp_name = "{0}_w1{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        im['normalize'][_tmp_name] = 0

        fac1 = T.dot(inpt1, _w)
        params.append(_w)

        _tmp = initweight(shape, variant=config["init"])
        _tmp_name = "{0}_w2{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        im['normalize'][_tmp_name] = 0

        fac2 = T.dot(inpt2, _w)
        params.append(_w)

        prod = fac1 * fac2

        _tmp = initweight(shape, variant=config["init"])
        _tmp_name = "{0}_w3{1}".format(tag, i)
        _w = theano.shared(value=_tmp, borrow=True, name=_tmp_name)
        params.append(_w)
        _tmp = np.zeros((shape[1],), dtype=theano.config.floatX)
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
    for comp in components:
        assert "type" in comp, "[PAR -- {0}] Subcomponent needs 'type'.".format(tag)
        typ = comp['type']
        
        _tag = comp['tag']
        comp['tag'] = "|".join([tag, _tag])

        comp['inpt'] = inpt
        typ(config=comp, params=params, im=im)
        
        assert "otpt" in comp, "[PAR -- {0}] Subcomponent needs 'otpt'.".format(tag)
        inpt = comp['otpt']
    
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
    # 1-d direction of orientation
    u = inpt[:, 2*dim:]

    _tmp = "{0}_kl_dg_g_mu".format(tag)
    im[_tmp] = mu
    _tmp = "{0}_kl_dg_g_log_var_inv".format(tag)
    im[_tmp] = log_var_inv
    _tmp = "{0}_kl_dg_g_u".format(tag)
    im[_tmp] = u
 
    mu_sq = mu * mu
    var_inv = T.exp(log_var_inv)

    # get log determinant
    # Du is D-1 * u in the paper
    Du = var_inv * u
    uDu = T.sum(Tu*Du, axis=1).dimshuffle(0, 'x')
    eta = 1./(uDu + 1)
    logDet = T.log(eta) + T.sum(T.log(var_inv), axis=1)

    # get trace (use some previous computations)
    Dusq = Du * Du
    # the minus here is newish
    trc =  T.sum(var_inv, axis=1) - eta*T.sum(Dusq, axis=1)

    # generate samples
    rng = T.shared_randomstreams.RandomStreams()
    # gaussian zero/one noise
    gzo = rng.normal(size=mu.shape, dtype=theano.config.floatX)
    trf = T.sum(u*T.sqrt(var_inv)*gzo, axis=1).dimshuffle(0, 'x')
    gzo = T.sqrt(var_inv)*gzo - (1-T.sqrt(eta))/uDu * trf * Du
    # Reparameterized latent variable
    z = mu + gzo
    _tmp = "{0}_kl_dg_g_z".format(tag)
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


vae_cost_ims[kl_dlap_lap] = ('kl_dlap_lap_mu', 'kl_dlap_laplog_var', 'kl_dlap_lap')


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
    cost = -(t*T.log(pred + 1e-4) + (1-t)*T.log(1-pred + 1e-4))
    cost = T.sum(cost, axis=1)
    cost = T.mean(cost)
    im['bern_xe'] = cost
    im['cost'] = im['cost'] + cost


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


def rmsprop(params, grads, **kwargs):
    """
    RMSprop.
    """
    eps = 10e-8
    lr = kwargs['lr']
    decay = kwargs['decay']
    mom = kwargs['momentum']

    print "[RMSprop] lr: {0}; decay: {1}, momentum: {2}".format(lr, decay, momentum)

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
