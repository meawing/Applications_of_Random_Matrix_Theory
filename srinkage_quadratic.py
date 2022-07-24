# This code is based on the code of
# Autor: Patrick Ledoit, MIT License

import numpy as np
import pandas as pd
import math

def ledoit_wolf_quad(cov_smp, T):
    """
        The quadratic-inverse shrinkage (QIS) estimator of a covariance matrix """
    """
    :param cov_smp: type np.ndarray. (NxN) sample covariance matrix
    :param T: number of observations (days)
    :return: type np.ndarray. Shrunk covariance matrix
    """

    N, T = cov_smp.shape[0], T
    # concentration ratio
    c = N / T 

    # ensure sample covariance matrix being symmetrical
    cov_smp = (cov_smp + cov_smp.T) / 2  
    
    # get the original eigenvalues and eigenvectors in sorted order
    e_val, e_vec = np.linalg.eigh(cov_smp)
    # reset negative values to 0
    e_val = e_val.real.clip(min=0)
    idx = e_val.argsort()[::1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    
    # Quadratic-Inverse Shrinkage estimator of the covariance matrix
    
    # smoothing parameter
    h = (min(c ** 2, 1 / c ** 2) ** 0.35) / N ** .35
    # inverse of (non-null) eigenvalues
    inve_val = 1 / e_val[max(1, N - T + 1) - 1:N]  
    dfl = pd.DataFrame()
    dfl['lambda'] = inve_val
    # like  1/lambda_j
    Lj = dfl[np.repeat(dfl.columns.values, min(N, T))] 
    # Reset column names
    Lj = pd.DataFrame(Lj.to_numpy())
    # like (1/lambda_j)-(1/lambda_i)
    Lj_i = Lj.subtract(Lj.T)                    
   
    # smoothed Stein shrinker
    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj) * h ** 2)).mean(axis = 0)  
    # its conjugate
    Htheta = Lj.multiply(Lj * h).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj) * h ** 2)).mean(axis = 0)  
    # its squared amplitude
    Atheta2 = theta ** 2 + Htheta ** 2                         
    
    # case where sample covariance matrix is not singular
    if N <= T:     
        # optimally shrunk eigenvalues
        delta = 1 / ((1 - c) ** 2 * inve_val + 2 * c * (1 - c) * inve_val * theta \
                      + c ** 2 * inve_val * Atheta2)    
        delta = delta.to_numpy()
    else:
        # shrinkage of null eigenvalues
        delta0 = 1 / ((c - 1) * np.mean(inve_val))                                                
        delta = np.repeat(delta0, N - T)
        delta = np.concatenate((delta, 1 / (inve_val * Atheta2)), axis=None)

    # preserve trace    
    deltaQIS = delta * (e_val.sum() / sum(delta))                  
    
    # reconstruct covariance matrix
    cov_shrunk = (e_vec @ np.diag(deltaQIS)) @ e_vec.T.conjugate()
    
    return cov_shrunk