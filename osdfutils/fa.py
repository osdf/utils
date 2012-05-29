"""
FA -- factor analysis
"""


import scipy.linalg as la
import numpy as np


SMALL = 1e-8


def fa_svd(data, hdims, psi=None, iters=100, eps=1e-1):
    """Factor Analysis.

    _data_ is an n times d datamatrix (i.e.
    samples are per row). It is assumed that
    _data_ has already mean zero!

    """
    n, d = data.shape

    # some constant terms
    nsqrt = np.sqrt(n)
    llconst = d*np.log(2*np.pi) + hdims

    if psi is None:
        psi = np.ones(d)

    var = np.var(data, 0)
    loglike = []
    tmp = -np.inf
    for i in xrange(iters):
        sqrt_psi = np.sqrt(psi) + SMALL
        Xtilde = data/(sqrt_psi * nsqrt)
        u, s, v = la.svd(Xtilde, full_matrices=False)
        v = v[:hdims]
        s = s**2
        # Use 'maximum' here to avoid sqrt problems.
        W = np.sqrt(np.maximum(s[:hdims] - 1, 0))[:, np.newaxis]*v
        W *= sqrt_psi
        ll = -n/2*(llconst + np.sum(np.log(s[:hdims])) + np.sum(s[hdims:]) + np.sum(np.log(psi)))
        loglike.append(ll)
        if ll - tmp < eps:
            break
        tmp = ll
        psi = var - np.sum(W**2, axis=0)
    return W, psi, loglike


def test():
    """
    """
    W0 = np.array([[-0.7024, 2.4048, 0.4068],
            [-0.4931, 0.0942, 0.0663],
            [0.6526, 1.2858, -0.4584],
            [1.3902, -0.0874, -0.0788],
            [1.3709, 0.2231, 0.7221]]).T
    h = np.array([-1.0891 ,1.1006 ,-1.4916 ,2.3505 ,-0.1924 ,-1.4023 ,-0.1774 ,0.2916 ,-0.8045 ,-0.2437 ,-1.1480 ,2.5855 ,-0.0825 ,-1.7947 ,0.1001 ,-0.6003 ,1.7119 ,-0.8396 ,0.9610 ,-1.9609 ,0.0326 ,1.5442 ,-0.7423 ,-0.6156 ,0.8886 ,-1.4224 ,-0.1961 ,0.1978 ,0.6966 ,0.2157 ,0.1049 ,-0.6669 ,-1.9330 ,0.8404 ,-0.5445 ,0.4900 ,-0.1941 ,1.3546 ,0.1240 ,-0.1977 ,0.5525 ,0.0859 ,-1.0616 ,0.7481 ,-0.7648 ,0.4882 ,1.4193 ,1.5877 ,0.8351 ,-1.1658 ,0.7223 ,0.1873 ,-0.4390 ,-0.8880 ,0.3035 ,0.7394 ,-2.1384 ,-1.0722 ,1.4367 ,-1.2078])
    h = h.reshape(3,20)
    h = h.T

    noise = np.array([-1.0642 ,-0.4446 ,0.3919 ,-0.3206 ,-1.0667 ,-1.5651 ,-0.7342 ,-0.2365 ,1.0001 ,-1.6702 ,0.3271 ,-0.9444 ,0.9111 ,0.2398 ,-0.0245 ,-0.0708 ,0.0799 ,-0.6912 ,0.8979 ,-0.5046 ,1.6035 ,-0.1559 ,-1.2507 ,0.0125 ,0.9337 ,-0.0845 ,-0.0308 ,2.0237 ,-1.6642 ,0.4716 ,1.0826 ,-1.3218 ,0.5946 ,-0.6904 ,-1.9488 ,-2.4863 ,-0.9485 ,0.4494 ,-0.1319 ,-1.2706 ,1.2347 ,0.2761 ,-0.9480 ,-3.0292 ,0.3503 ,1.6039 ,0.2323 ,-2.2584 ,-0.5900 ,-1.2128 ,1.0061 ,0.9248 ,0.3502 ,-0.6516 ,1.0205 ,0.5812 ,0.4115 ,0.1006 ,-0.1472 ,-0.3826 ,-0.2296 ,-0.2612 ,-0.7411 ,-0.4570 ,-0.0290 ,0.0983 ,0.4264 ,2.2294 ,-0.2781 ,0.0662 ,-0.6509 ,0.0000 ,1.2503 ,1.1921 ,0.8617 ,-2.1924 ,0.6770 ,0.8261 ,1.0078 ,0.6487 ,-1.5062 ,0.4434 ,-0.5078 ,1.2424 ,0.1825 ,0.0414 ,-0.3728 ,0.3376 ,0.4227 ,0.6524 ,0.2571 ,-0.0549 ,0.9298 ,-1.6118 ,0.0012 ,-2.3193 ,0.8577 ,0.5362 ,-2.1237 ,0.8257])
    noise = noise.reshape(5, 20)
    noise = noise.T

    X = np.dot(h, W0) + 0.1 * noise;
    data = X - X.mean(axis=0)
    W, psi, ll = fa_svd(data , hdims=3, iters=50, eps=1e-1);
    print W
    print psi
    print ll
    print 'Sample Covariance', np.cov(data, rowvar=0, bias=1)
    print 'Model Covariance', np.dot(W.T, W) + np.diag(psi)


#%function [F,diagPsi,m,loglik]=FA(X,H,varargin)
#%%FA Factor Analysis
#%% [F,diagPsi,m,loglik]=FA(X,H,<opts>)
#%%
#%% No post rotation of the factors F is applied
#%%
#%% Inputs:
#%% X is the D x N data matrix with each column being a D-dimensional datapoint
#%% H is the number of factors
#%% opts.init.diagPsi: initial D dimensional vector describing the diagonal covariance matrix
#%% opts.init.F : initial Factor loading matrix (ignored if opts.init.diagPsi exists)
#%% opts.maxit: maximum number of iterations (default is 100)
#%% opts.tol : change in log likelihood termination criterion (default is 10e-5)
#%% opts.plotprogress: set to 1 to display log likelihood (default is 0)
#%%
#%% Outputs:
#%% F : factor loading matrix
#%% diagPsi : diagonal entries of the covariance
#%% m : mean of the data
#%% loglik: log likelihood of the data
#%for loop=1:opts.maxit
#%    diagSqrtPsi = sqrt(diagPsi)+10e-10;
#%    Xtilde = diag(1./diagSqrtPsi)*X/sqrt(N);
#%    [U,LambdaTilde] = svd(Xtilde,0);
#%    Uh = U(:,1:H); diagLambda=diag(LambdaTilde).^2;
#%    F = diag(diagSqrtPsi)*Uh*diag(sqrt(max(diagLambda(1:H)-1,0)));
#%    loglik=-0.5*N*(sum(log(diagLambda(1:H)))+H+sum(diagLambda(H+1:end))+V*log(2*pi)+sum(log(diagPsi)));
#%    logpV(loop) = loglik;
#%    if loop>1
#%        if  logpV(loop) - logpV(loop-1) < opts.tol; break; end
#%    end
#%    if opts.plotprogress;    plot(logpV,'-o'); title('Log likelihood'); drawnow; end
#%    diagPsi=diagS-sum(F.^2,2);
#%end
