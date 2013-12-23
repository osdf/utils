import numpy as np
import scipy.signal as scsig


import misc


def score(weights, structure, inputs, targets,
        predict=False, error=False, **kwargs):
    """
    """
    layers = structure["layers"]
    pools = structure["pools"]
    activs = structure["activs"]
    
    z = inputs
    max_idx = []
    
    if error:
        hiddens = []
        maxima = []
        hiddens.append(z)

    idx = 0
    for l, p, A in zip(layers, pools, activs):
        # weights in a layer: feature maps * in_channels * dx * dy
        idy = idx + l[0]*l[1]*l[2]*l[3]*l[4]
        filters = weights[idx:idy].reshape(l)
        z = conv(z, filters)
        z, max_idx = max_pool(z, p)
        idx = idy+l[0]
        z = A(z + weights[idy:idx].reshape(1, l[0], 1, 1, 1))
        if error:
            hiddens.append(z)
            maxima.append(max_idx)

    if error:
        structure['hiddens'] = hiddens
        structure['maxima'] = maxima

    # reshape z
    z = np.reshape(z, (z.shape[0], z.shape[1]*z.shape[2]*z.shape[3]*z.shape[4]))
    sc = structure['score']
    return sc(z, targets, predict=predict, error=error, addon=0)


def score_grad(weights, structure, inputs, targets, **params):
    """
    """
    g = np.zeros(weights.shape, dtype=weights.dtype)

    # forward pass through model,
    # need 'error' signal at the end.
    sc, delta = score(weights, inputs=inputs, targets=targets,
            structure=structure, predict=False, error=True, **params)
    layers = structure["layers"]
    activs = structure["activs"]
    pools = structure['pools']
    hiddens = structure["hiddens"]
    maxima = structure['maxima']

    delta = np.reshape(delta, (delta.shape[0], delta.shape[1], 1, 1, 1))
    tmp = hiddens[-1]
    #print 'shapes', delta.shape, tmp.shape
    idy = 0
    idx = 0
    for l, p, A, h, m in reversed(zip(layers, pools, activs, hiddens[:-1], maxima)):
        # tmp are values _after_ applying 
        # activation A to matrix-vector product
        # Compute dE/da (small a -> before A is applied)
        dE_da = delta * (misc.Dtable[A](tmp))
        idx = idy + l[0]
        # gradient for biases
        if idy == 0:
            g[-idx:] = dE_da.sum(axis=(0, 2, 3, 4))
        else:
            g[-idx:-idy] = dE_da.sum(axis=(0, 2, 3, 4))
        
        idy = idx + l[0] * l[1] * l[2] * l[3] * l[4]
        delta = max_pool_bp(delta, m, p)
        # gradient for weights in layer l
        g[-idy:-idx] = conv_grad(h, delta).ravel()
        # backprop delta
        delta = conv_bp(delta, weights[-idy:-idx].reshape(l))
        tmp = h
    g += 2*structure["l2"]*weights
    # clean up structure
    del structure["hiddens"]
    return sc, g


def grad(weights, structure, inputs, targets, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, targets, **params)
    return g


def init_weights(structure, var=0.01):
    """
    """
    size=0
    layers = structure["layers"]
    # determine number of total weights
    for l in layers:
        size += l[0]*l[1]*l[2]*l[3]*l[4] + l[0]
    weights = np.zeros(size)
    # init weights. biases are 0. 
    idx = 0
    for l in layers:
        weights[idx:idx+l[0]*l[1]*l[2]*l[3]*l[4]] = var*np.random.randn(l[0]*l[1]*l[2]*l[3]*l[4])
        idx += l[0] * l[1] * l[2] * l[3] * l[4] + l[0]
    return weights


def check_the_grad(eps=1e-8, verbose=False):
    """
    """
    from opt import check_grad
    from losses import ssd

    structure = {}
    structure['layers'] = [(3, 1, 2, 2, 2), (1, 3, 2, 2, 2)]
    structure['pools'] = [(2, 2, 2), (2, 2, 2)]
    structure['activs'] = [misc.identy, misc.identy]
    structure['score'] = ssd
    structure['l2'] = 0.

    weights = init_weights(structure)
    inpts = np.random.randn(2, 1, 7, 7, 7)
    trgts = np.random.randn(2).reshape(2, 1)
    cg = dict()
    cg["inputs"] = inpts
    cg["targets"] = trgts
    cg["structure"] = structure
    #
    #grad(weights, structure, inpts, trgts)
    delta = check_grad(score, grad, weights, cg, eps=eps, verbose=verbose)


def conv(inputs, filters):
    batch, channels, rows, cols, depth = inputs.shape
    n_fltr, ch_fltr, row_fltr, col_fltr, depth_fltr = filters.shape

    drows = rows - row_fltr + 1
    dcols = cols - col_fltr + 1
    ddepth = depth - depth_fltr + 1
    result = np.zeros((batch, n_fltr, drows, dcols, ddepth))
    
    for b in range(batch):
        for n in range(n_fltr):
            for c in range(channels):
                for i in range(drows):
                    for j in range(dcols):
                        for k in range(ddepth):
                            for xx in range(row_fltr):
                                for yy in range(col_fltr):
                                    for zz in range(depth_fltr):
                                        val = inputs[b, c, i+xx, j+yy, k+zz]
                                        fval = filters[n, c, xx, yy, zz]
                                        result[b, n, i, j, k] += val * fval
    return result


def conv_bp(delta, filters):
    """
    """
    batch, channels, rows, cols, depth = delta.shape
    n_fltr, ch_fltr, row_fltr, col_fltr, depth_fltr = filters.shape

    delta_down = np.zeros((batch, ch_fltr, rows+row_fltr-1, cols+col_fltr-1, depth+depth_fltr-1))

    for b in range(batch):
        for c in range(ch_fltr):
            for n in range(n_fltr):
                for i in range(-row_fltr+1,rows):
                    for j in range(-col_fltr+1,cols):
                        for k in range(-depth_fltr+1, depth):
                            for xx in range(row_fltr):
                                for yy in range(col_fltr):
                                    for zz in range(depth_fltr):
                                        if (i+xx >= 0) and (j+yy >= 0) and (k+zz >= 0) and (i+xx<rows) and (j+yy<cols) and (k+zz<depth):
                                            val = delta[b, n, i+xx, j+yy, k+zz]
                                            fval = filters[n, c, row_fltr-xx-1, col_fltr-yy-1, depth_fltr-zz-1]
                                            delta_down[b, c, i+row_fltr-1, j+col_fltr-1, k+depth_fltr-1] += val*fval
    return delta_down


def conv_grad(inputs, deltas):
    """
    """
    batch, channels, rows, cols, depth = inputs.shape
    batch, fmaps, drows, dcols, ddepth = deltas.shape
    dr = rows - drows + 1
    dc = cols - dcols + 1
    dd = depth - ddepth + 1

    grad = np.zeros((fmaps, channels, dr, dc, dd))
    for f in range(fmaps):
        for c in range(channels):
            for i in range(dr):
                for j in range(dc):
                    for k in range(dd):
                        for b in range(batch):
                             for xx in range(drows):
                                 for yy in range(dcols):
                                     for zz in range(ddepth):
                                        val = inputs[b, c, i+xx, j+yy, k+zz]
                                        fval = deltas[b, f, xx, yy, zz]
                                        grad[f, c, i, j, k] += val * fval
    return grad


def max_pool(inputs, pool_sz):
    """
    """
    mpr, mpc, mpd = pool_sz
    batch, maps, rows, cols, depth = inputs.shape
    maxima = -np.inf * np.ones((batch, maps, rows//mpr, cols//mpc, depth//mpd))
    idx = np.zeros(maxima.shape, dtype=np.int)

    for b in range(batch):
        for m in range(maps):
            for r in range(rows):
                for c in range(cols):
                    for d in range(depth):
                        if inputs[b, m, r, c, d] > maxima[b, m, r//mpr, c//mpc, d//mpd]:
                            maxima[b, m, r//mpr, c//mpc, d//mpd] = inputs[b, m, r, c, d]
                            idx[b, m, r//mpr, c//mpc, d//mpd] = r%mpr * (mpc*mpd) + c%mpc * mpd + d%mpd
    return maxima, idx


def max_pool_bp(delta, idx, pool_sz):
    """
    """
    mpr, mpc, mpd = pool_sz
    batch, maps, rows, cols, depth = delta.shape
    delta_down = np.zeros((batch, maps, mpr*rows, mpc*cols, mpd*depth))
    for b in range(batch):
        for m in range(maps):
            for r in range(rows):
                for c in range(cols):
                    for d in range(depth):
                        rc = idx[b, m, r, c, d]
                        rr = rc//(mpc*mpd)
                        tmp = rc%(mpc*mpd)
                        cc = tmp//mpd
                        dd = tmp%mpd
                        delta_down[b, m, r*mpr+rr, c*mpc+cc, d*mpd+dd] = delta[b, m, r, c, d]
    return delta_down

def check_deriv():
    """
    """
    from losses import ssd
    eps = 1e-6
    structure = {}
    structure['layers'] = [(1, 1, 2, 2, 2), (1, 1, 2, 2, 2)]
    structure['pools'] = [(2, 2, 2), (2, 2, 2)]
    structure['activs'] = [misc.identy, misc.identy]
    structure['score'] = ssd
    structure['l2'] = 0.

    weights = init_weights(structure)
    trgts = np.asarray([[1]])

    inpts = np.asarray([
         [[
             [[1,1,1,1,1,1,1],[1,1,0,0,0,0,0],[1,1,1,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]],
             [[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]],
             [[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]],
             [[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]],
             [[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]],
             [[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]],
             [[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]]
         ]]])
    deriv = np.zeros(inpts.shape)
    hlp = np.zeros(inpts.shape)
    print deriv.shape
    a, b, c, d, e = inpts.shape
    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    for m in range(e):
                        f1 = score(weights, structure, inpts, trgts)
                        f2 = score(weights, structure, inpts, trgts)
                        deriv[i,j,k,l,m] = (f1-f2)/(2*eps)
    print deriv
    score_grad(weights, structure, inpts, trgts)
