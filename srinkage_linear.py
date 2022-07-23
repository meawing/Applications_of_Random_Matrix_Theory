import numpy as np

def ledoit_wolf_lin(X, d=False):
    """
        A linear shrinkage of covariance matrix towards constant-correlation matrix:
        the target preserves the variances of the sample covariance matrix
        all the correlation coefficients of the target are the same """
    """   
    :param X: type np.ndarray. (TxN) matrix, where T is the number of observations (days), N is the number of features
    :param d: type bool. False means no data demeaning took place
    :return: type np.ndarray. Shrunk covariance matrix
    """
    T, N = X.shape 
    
    if not d:
        X_mean = np.mean(X, axis=0) 
        X = X - X_mean

    # sample covariance matrix
    cov_smp = X.T @ X / T
    
    # shrinkage target
    var_smp = np.diag(cov_smp).reshape(-1, 1)
    std_smp = np.sqrt(var_smp)
    corr_avg = ((cov_smp / (std_smp @ std_smp.T)).sum() - N)/(N * (N - 1))
    target = corr_avg * (std_smp @ std_smp.T)
    np.fill_diagonal(target, var_smp)
    
    # the parameter pi
    Y = X ** 2
    phi_ = (Y.T @ Y) / T - cov_smp ** 2
    phi = phi_.sum()    

    # the parameter rho
    rho_diag =  np.diag(phi_).sum()
    rho_off_diagonal = ((X ** 3).T @ X) / T - var_smp * cov_smp
    np.fill_diagonal(rho_off_diagonal, 0)
    rho = np.diag(phi_).sum() + corr_avg * (1 / std_smp @ std_smp.T * rho_off_diagonal).sum()

    # the parameter gamma
    gamma = np.linalg.norm(cov_smp - target, ord = 'fro') ** 2
    
    # shrinkage constant
    delta = max(0, min(1, (phi - rho) / gamma / T))    
    
    # compute shrinkage estimator
    cov_shrunk = delta * target + (1 - delta) * cov_smp
    
    return cov_shrunk