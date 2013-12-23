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
        idy = idx + l[0]*l[1]*l[2]*l[3]
        filters = weights[idx:idy].reshape((l[0], l[1], l[2], l[3]))
        z = conv(z, filters)
        z, max_idx = max_pool(z, p)
        idx = idy+l[0]
        z = A(z + weights[idy:idx].reshape(1, l[0], 1, 1))
        if error:
            hiddens.append(z)
            maxima.append(max_idx)

    if error:
        structure['hiddens'] = hiddens
        structure['maxima'] = maxima

    # reshape z
    z = np.reshape(z, (z.shape[0], z.shape[1]*z.shape[2]*z.shape[3]))
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

    delta = np.reshape(delta, (delta.shape[0], delta.shape[1], 1, 1))
    tmp = hiddens[-1]
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
            g[-idx:] = dE_da.sum(axis=(0, 2, 3))
        else:
            g[-idx:-idy] = dE_da.sum(axis=(0, 2, 3))
        
        idy = idx + l[0] * l[1] * l[2] * l[3]
        delta = max_pool_bp(delta, m, p)
        # gradient for weights in layer l
        g[-idy:-idx] = conv_grad(h, delta).ravel()
        # backprop delta
        delta = conv_bp(delta, weights[-idy:-idx].reshape(l[0], l[1], l[2], l[3]))
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
        size += l[0]*l[1]*l[2]*l[3] + l[0]
    weights = np.zeros(size)
    # init weights. biases are 0. 
    idx = 0
    for l in layers:
        weights[idx:idx+l[0]*l[1]*l[2]*l[3]] = 0.5+0*var*np.random.randn(l[0]*l[1]*l[2]*l[3])
        idx += l[0] * l[1] * l[2] * l[3] + l[0]
    return weights


def check_the_grad(eps=1e-6, verbose=False):
    """
    """
    from opt import check_grad
    from losses import ssd

    structure = {}
    structure['layers'] = [(1, 1, 2, 2), (1, 1, 2, 2)]
    structure['pools'] = [(2, 2), (2, 2)]
    structure['activs'] = [misc.identy, misc.identy]
    structure['score'] = ssd
    structure['l2'] = 0.

    weights = init_weights(structure)
    inpts = np.random.randn(2, 1, 7, 7)
    trgts = np.random.randn(2).reshape(2, 1)
    cg = dict()
    cg["inputs"] = inpts
    cg["targets"] = trgts
    cg["structure"] = structure
    #
    delta = check_grad(score, grad, weights, cg, eps=eps, verbose=verbose)


def conv(inputs, filters):
    batch, channels, rows, cols = inputs.shape
    n_fltr, ch_fltr, row_fltr, col_fltr = filters.shape

    drows = rows - row_fltr + 1
    dcols = cols - col_fltr + 1
    result = np.zeros((batch, n_fltr, drows, dcols))
    
    for b in range(batch):
        for n in range(n_fltr):
            for c in range(channels):
                for i in range(drows):
                    for j in range(dcols):
                        for xx in range(row_fltr):
                            for yy in range(col_fltr):
                                val = inputs[b, c, i+xx, j+yy]
                                fval = filters[n, c, xx, yy]
                                result[b, n, i, j] += val * fval
    return result


def conv_bp(delta, filters):
    """
    """
    batch, channels, rows, cols = delta.shape
    n_fltr, ch_fltr, row_fltr, col_fltr = filters.shape

    delta_down = np.zeros((batch, ch_fltr, rows+row_fltr-1, cols+col_fltr-1))

    for b in range(batch):
        for c in range(ch_fltr):
            for n in range(n_fltr):
                for i in range(-row_fltr+1,rows):
                    for j in range(-col_fltr+1,cols):
                        for xx in range(row_fltr):
                            for yy in range(col_fltr):
                                if (i+xx >= 0) and (j+yy >= 0) and (i+xx<rows) and (j+yy<cols):
                                    val = delta[b, n, i+xx, j+yy]
                                    fval = filters[n, c, row_fltr-xx-1, col_fltr-yy-1]
                                    delta_down[b, c, i+row_fltr-1, j+col_fltr-1] += val*fval
    return delta_down


def conv_grad(inputs, deltas):
    """
    """
    batch, channels, rows, cols = inputs.shape
    batch, fmaps, drows, dcols = deltas.shape
    dr = rows - drows + 1
    dc = cols - dcols + 1

    grad = np.zeros((fmaps, channels, dr, dc))
    for f in range(fmaps):
        for c in range(channels):
            for i in range(dr):
                for j in range(dc):
                    for b in range(batch):
                        for xx in range(drows):
                            for yy in range(dcols):
                                val = inputs[b, c, i+xx, j+yy]
                                fval = deltas[b, f, xx, yy]
                                grad[f, c, i, j] += val * fval
    return grad


def check_grad(inpts, filters):
    """
    """
    fmaps, channels, rows, cols = filters.shape
    holl = np.zeros(filters.shape)
    eps = 10e-6
    ngrad = np.zeros(filters.shape)

    for f in range(fmaps):
        for c in range(channels):
            for r in range(rows):
                for cl in range(cols):
                    holl[f, c, r, cl] = eps
                    f1 = (conv(inpts, filters+holl) - 1)**2
                    f2 = (conv(inpts, filters-holl) - 1)**2
                    ngrad[f, c, r, cl] = (f1.sum() - f2.sum())/(2*eps)
                    holl[f, c, r, cl] = 0.
    return ngrad


def conv_scpy(inputs, filters, res):
    """
    """
    res[:] = scsig.correlate(inputs, filters, mode='valid')


def fbcorr(imgs, filters, output):
    n_imgs, n_rows, n_cols, n_channels = imgs.shape
    n_filters, height, width, n_ch2 = filters.shape

    for ii in range(n_imgs):
        for rr in range(n_rows - height + 1):
            for cc in range(n_cols - width + 1):
                for hh in xrange(height):
                    for ww in xrange(width):
                        for jj in range(n_channels):
                            for ff in range(n_filters):
                                imgval = imgs[ii, rr + hh, cc + ww, jj]
                                filterval = filters[ff, hh, ww, jj]
                                output[ii, rr, cc, ff] += imgval * filterval


def max_pool(inputs, pool_sz):
    """
    """
    mpr, mpc = pool_sz
    batch, maps, rows, cols = inputs.shape
    maxima = -np.inf * np.ones((batch, maps, rows//mpr, cols//mpc))
    idx = np.zeros(maxima.shape, dtype=np.int)

    for b in range(batch):
        for m in range(maps):
            for r in range(rows):
                for c in range(cols):
                    if inputs[b, m, r, c] > maxima[b, m, r//mpr, c//mpc]:
                        maxima[b, m, r//mpr, c//mpc] = inputs[b, m, r, c]
                        idx[b, m, r//mpr, c//mpc] = r%mpr * mpc + c%mpc
    return maxima, idx


def max_pool_bp(delta, idx, pool_sz):
    """
    """
    mpr, mpc = pool_sz
    batch, maps, rows, cols = delta.shape
    delta_down = np.zeros((batch, maps, mpr*rows, mpc*cols))
    for b in range(batch):
        for m in range(maps):
            for r in range(rows):
                for c in range(cols):
                    rc = idx[b, m, r, c]
                    rr = rc//mpc
                    cc = rc%mpc
                    delta_down[b, m, r*mpr+rr, c*mpc+cc] = delta[b, m, r, c]
    return delta_down


def main():
    imgs = np.random.randn(10, 64, 64, 3)
    filt = np.random.randn(6, 5, 5, 3)
    output = np.zeros((10, 60, 60, 6))

    fbcorr(imgs, filt, output)
