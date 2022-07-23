import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import warnings


def marcenko_pastur(q, var, pts=1000):
    """
        Marcenko-Pastur pdf """
    """
    :param q: = T / N
    :param var: the variance of underlying process generating the observations
    :param pts: the number of points for linear span
    """    
    # upper and lower bound for eigenvalues
    e_min, e_max = var * (1 - (1. / q) ** .5) **2, var * (1 + (1. / q) ** .5) ** 2
    e_span = np.linspace(e_min, e_max, pts).flatten()
    
    # Marcenko – Pastur probability density function
    pdf = q / (2 * np.pi * var * e_span) * ((e_max - e_span) * (e_span - e_min)) ** .5
    return pdf, e_span
   
    
def kernel(obs, bwidth=.25, kernel='gaussian', x=None):
    """
        Fit kernel to a series of observations, and derive the probability """
    """
    :param obs: the array of values to fit KDE to
    :param x: the array of values on which the fit KDE will be evaluated
    """
    obs = obs.reshape(-1,1) if len(obs.shape) == 1 else obs
    
    kde = KernelDensity(kernel=kernel, bandwidth=bwidth).fit(obs)

    x = np.unique(obs).reshape(-1, 1) if x is None else x
    x = x.reshape(-1, 1) if len(x.shape) == 1 else x
    
    log_prob = kde.score_samples(x)
    pdf = np.exp(log_prob)
    
    return pdf


def fit(var, e_val, q, bwidth, pts=1000):
    """
        Compute the sum of squared errors between pdf series
    """        
    # theoretical pdf
    pdf_mp, span = marcenko_pastur(q, var, pts)
    
    # empirical pdf
    pdf_kernel = kernel(e_val, bwidth, x=span)
            
    return ((pdf_mp - pdf_kernel) ** 2).sum()


def laloux(cov_smp, T, bwidth=.01):
    """ Denoising covariance matrix by shrinking eigenvalues to a trace-preserving constant
        that fall within determined by Marchenko-Pastur distribution threshold  """
    """
    :param cov_smp: type np.ndarray. (NxN) sample covariance matrix
    :param T: number of observations (days)
    :param bwidth: bandwidth of the kernel
    :return: type np.ndarray. Denoised covariance matrix
    """

    N = cov_smp.shape[0]
    q = T / N
    if q < 1:
        warnings.warn("""It is assumed that the number of samples
                        T must be higher than the number of features N""", UserWarning)    
        
    # scale sample covariance to correlation matrix
    std_smp = np.sqrt(np.diag(cov_smp).reshape(-1, 1))
    corr_smp = cov_smp / (std_smp @ std_smp.T)
    
    # assuming the unite variance of underlying process since the correlation has ones on the diagonal
    var = 1
    
    # return the original eigenvalues and eigenvectors
    e_val, e_vec = np.linalg.eigh(corr_smp)
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    e_val = np.diagflat(e_val)
    
    # find max random eigen value by fitting Marcenko-Pastur dist 
    out = minimize(lambda *x: fit(*x), .5, args=(np.diag(e_val), q, bwidth), bounds=((1E-5,1-1E-5),))    
    var_fit = out['x'][0] if out['success'] else var
    e_max = var_fit * (1 + (1. / q) ** .5) ** 2
    
    # return the number of non-random factors (num of eigenvalues above the threshhold)   
    n_factors = e_val.shape[0] - np.diag(e_val)[::-1].searchsorted(e_max)
    
    # remove noise from corr by trace preserving change of random eigenvalues
    e_val_= np.diag(e_val).copy()
    e_val_[n_factors:] = e_val_[n_factors:].sum() / float(e_val_.shape[0] - n_factors)
    e_val_= np.diag(e_val_)
    corr = np.dot(e_vec, e_val_).dot(e_vec.T)
    
    # rescale denoised correlation matrix back to covariance
    cov_den = corr * (std_smp @ std_smp.T)
    
    return cov_den