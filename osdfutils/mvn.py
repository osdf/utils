import numpy as np
import scipy.linalg as la


def contour_2d(mu, cov=None, prec=None, 
        n=100, radius=[1, np.sqrt(6)]):
    """
    Assuming a bivariate normal
    distribution, draw contours,
    given 'radius' information.

    Note:
    sqrt(6) covers roughly 95% of probability
    mass (in 2d), given the fact that the mahalanobis
    distribution is chi-squared distributed.
    """
    mu = mu.reshape(2,1) 
    t = np.linspace(0, 2*np.pi, n)
    circle = np.array([np.cos(t), np.sin(t)])
    if prec is None:
        L = la.cholesky(cov)
        ellipse = np.dot(L, circle)
    else:
        L = la.cholesky(prec)
        ellipse = la.solve_triangular(L, circle)
        # FIXME: not correct yet
    plots = {}
    for r in radius:
        plots[r] = (r*ellipse[0,:] + mu[0], r*ellipse[1,:] + mu[1])
    return plots
    # From here: only plotting stuff
    #ax = pylab.subplot(111)
    #for r in radius:
    #    ax.plot(r*ellipse[0,:] + mu[0], r*ellipse[1,:] + mu[1])
    #ax.plot(mu[0], mu[1], mu_color + 'o')
    #ax.set_aspect('equal')


def sample(n, mu, cov):
    """
    Given _mu_ and covariance _cov_,
    produce _n_ samples from this
    gaussian. 
    
    Returns a matrix with 
    samples per row.
    
    Uses cholesky decomposition.

    For plotting these samples, if in 2d:
    pylab.plot(samples[:,0], samples[:,1], 'o')
    """
    (_, d) = mu.shape
    samples = np.random.randn(n, d)
    L = la.cholesky(cov)
    samples = np.dot(samples, L) + mu
    return samples


def mle(samples):
    """
    Compute maximum likelihood 
    estimates for mu and covariance.
    _samples_ is the design matrix,
    i.e. rowwise data vectors. 
    """
    mu = np.mean(samples, axis=0)[:, np.newaxis]
    cov = np.cov(samples, rowvar=0)
    return mu, cov


def bayes_mean(samples, prec, mu_0=None, prec_0=None):
    """
    Infer the posterior gaussian distribution
    for mean mu. Assume that _prec_ is the
    known precision of the distribution of
    _samples_ (the observation noise). 
    _mu_0_ and _cov_0_ define the
    Gaussian prior over mu. If _prec_0_ is None,
    a noninformative prior (i.e. prior 'precision' = 0*Id)
    is assumed.

    _samples_ is the design matrix (rowwise observations).

    Note: This is a special case of bayes_measure below.

    Returns posterior mean (a column vector) and
    posterior precision.
    """
    (n, _) = samples.shape
    if prec_0 is None:
        post_mean = samples.mean(axis=0)
        post_prec = n * prec 
    else:
        post_prec = prec_0 + n * prec 
        mean_a = np.dot(prec, np.atleast_2d(samples.sum(axis=0)).T)
        mean_b = np.dot(prec_0, mu_0)
        post_mean = la.solve(post_prec, (mean_a + mean_b), sym_pos=True)
    return post_mean.reshape(-1,1) , post_prec


def log_eval(samples, mu, cov):
    """
    Compute the log likelihood of _samples_,
    given _mu_ and _cov_.

    _samples_ is the design matrix (number of samples x dim).
    _mu_ is the expected value of the gaussian, a column.
    _cov_ is the covariance matrix of the gaussian.

    Returns an array of respective log likelihoods.
    """
    (n, d) = samples.shape
    log_const = -d/2. * np.log(2*np.pi)
    chol = la.cholesky(cov, lower=True)
    # determinant: product of _squared_ diagonals
    # of cholesky factor -> log produces sum,
    # however: factor _2_ (from squares) is cancelled
    # with square root from definition of gauss pdf.
    logdet = np.sum(np.log(np.diag(chol)))
    mhlb = la.solve_triangular(chol, (samples - mu.T).T, lower=True).T
    # re-note: no 0.5 before logdet, cancels with 2* from logdet
    return log_const - logdet - 0.5 * np.sum(mhlb**2, axis=1)


def log_eval_prec(samples, mu, prec):
    """
    Compute log likelihood of _samples,
    given _mu_ and precision _prec_.

    cf. log_eval above
    """
    (n, d) = samples.shape
    log_const = -d/2. * np.log(2*np.pi)
    chol = la.cholesky(prec, lower=True)
    # Use relation between determinan of A and A^{-1}
    logdet = -np.sum(np.log(np.diag(chol)))
    mhlb = np.dot(samples - mu.T, chol)  
    return log_const - logdet - 0.5 * np.sum(mhlb**2, axis=1)


def bayes_measure(mean_0, prec_0, sensors, samples):
    """
    Prior information about an entity can 
    be described by a gaussian with
    _mean_ and precision _prec_. _sensors_ are
    'gaussian' sensors for this entity. These
    sensors produced several _samples_. Infer the
    parameters for the posterior hypothesis of the
    entity (it will be distributed according
    to a gaussian).

    _sensors_ is a dictionary of several sensors.
    Every sensor is desribed by precision (i.e.
    there is an error model for the sensors). The
    mean of these sensors is of course the entity
    in question (otherwise, get better sensors :) ).
    _samples_ is also a dictionry containing for
    every sensor in _sensors_ a set of measurements.
    The set of measurements has the usual form of
    a design matrix per sensor (i.e. rowwise observations).

    Specifically:
    - _sensors_ looks like: {"sensor1": prec_1, "sensors2": prec_2, ...}
    - _samples_ looks like: {"sensor1": obs1, "sensor2": obs2, ...}

    Note: All involved matrices/vectors must have same type (float or double)
    Return the posterior mean and precision of
    the measured entity.

    Tests: 
    Prior information about entity: Normal(0, 0.1I_2), a 2d gaussian.
    True position: (0.5, 0.5). Have one sensor, with Cov = 0.1[[2, 1], [1, 1]].
    Sample 10 points from N(true position, Cov), and run inference.
    prior_mu = np.array([[0],[0]])
    prior_prec = 10.*np.array([[1,0],[0,1]])
    mu = np.array([[0.5],[0.5]])
    cov = 0.1*np.array([[2,1],[1,1,]])
    prec = la.inv(cov) #yes, you shouldn't do it this way!
    smpls = sample(10, mu, cov)
    sensors = {"s1": prec}
    samples = {"s1": smpls}
    pm, pp = bayes_measure(prior_mu, prior_prec, sensors, samples)

    Use previous prior information and true position. Now, we
    have one sensor that is only reliable in the x-direction, and
    one that is only reliable in the y-direction. Each sensor only
    measures once:
    prior_mu = np.array([[0],[0]])
    prior_prec = 10.*np.array([[1,0],[0,1]])
    mu = np.array([[0.5],[0.5]])
    cov1 = 0.01 * np.array([[10, 1], [1,1]])
    cov2 = 0.01 * np.array([[1, 1,], [1, 10]])
    s1 = sample(1, mu, cov1)
    s2 = sampel(1, mu, cov2)
    prec1 = la.inv(cov1)
    prec2 = la.inv(cov2)
    sensors = {"s1": prec1, "s2": prec2}
    samples = {"s1": s1, "s2": s2}
    pm, pp = bayes_measure(prior_mu, prior_prec, sensors, samples)
    """
    post_prec = prec_0.copy()
    post_mean = np.dot(prec_0, mean_0)
    for s in sensors.keys():
        post_prec += samples[s].shape[0]*sensors[s]
        post_mean += np.dot(sensors[s], np.atleast_2d(samples[s].sum(axis=0)).T)
    post_mean = la.solve(post_prec, post_mean, sym_pos=True)
    return post_mean, post_prec
