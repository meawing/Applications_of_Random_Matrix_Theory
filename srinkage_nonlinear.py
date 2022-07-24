# This code is based on pyRMT source code
# Autors: Gregory Giecold and Lionel Ouaknin, MIT License

import numpy as np

def pool_adjacent_violators(y):
    """
        Monotonic smoothing of y """

    y = np.asarray(y)
    
    assert y.ndim == 1
      
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    
    while True:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        violator = np.where(deriv < 0)[0]
        start = lvlsets[violator[0], 0]
        last = lvlsets[violator[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
            
    return v


def ledoit_wolf_nonlin(cov_smp, T):
    """
        A non-linear shrinkage estimator of a covariance marix
        based on the spectral distribution of its eigenvalues and that of its Hilbert Tranform """
    """
    :cov_smp: type np.ndarray. (NxN) sample covariance matrix
    :param T: number of observations (days)
    :return: type np.ndarray. Shrunk covariance matrix
    """

    N = cov_smp.shape[0]
    q = N / T

    # get the original eigenvalues and eigenvectors in sorted order
    e_val, e_vec = np.linalg.eigh(cov_smp)
    idx = e_val.argsort()[::1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    # compute analytical nonlinear shrinkage kernel formula
    
    lambda_ = e_val[max(0, N - T):].T  
    
    h = T ** (- .35)
    h_squared = h ** 2    
    
    L = np.tile(lambda_, (N, 1)).T
    Lt = L.T
    square_Lt = h_squared * (Lt ** 2)
    
    zeros = np.zeros((N, N))
    
    tmp = np.sqrt(np.maximum(4 * square_Lt - (L - Lt) ** 2, zeros)) / (2 * np.pi * square_Lt)
    f_tilde = np.mean(tmp, axis=0)
    
    tmp = np.sign(L - Lt) * np.sqrt(np.maximum((L - Lt) ** 2 - 4 * square_Lt, zeros)) - L + Lt 
    tmp /= 2 * np.pi * square_Lt
    Hf_tilde = np.mean(tmp, axis=1)
    
    if N <= T:
        tmp = (np.pi * q * lambda_ * f_tilde) ** 2
        tmp += (1 - q - np.pi * q * lambda_ * Hf_tilde) ** 2
        d_tilde = lambda_ / tmp
    else:
        Hf_tilde_0 = (1 - np.sqrt(1 - 4 * h_squared)) / (2 * np.pi * h_squared) * np.mean(1. / lambda_)
        d_tilde_0 = 1 / (np.pi * (N - T) / T * Hf_tilde_0)
        d_tilde_1 = lambda_ / ((np.pi ** 2) * (lambda_ ** 2) * (f_tilde ** 2 + Hf_tilde ** 2))
        d_tilde = np.concatenate(np.dot(d_tilde_0, np.ones(N - T, 1, np.float)), d_tilde_1)
        
    d_hats = pool_adjacent_violators(d_tilde)    
    cov_srunk = np.dot(e_vec, (np.tile(d_hats, (N, 1)).T * e_vec.T))
    
    return cov_srunk
