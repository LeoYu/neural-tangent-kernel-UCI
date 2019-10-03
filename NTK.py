import math
import numpy as np

# return an array K of size (d_max, d_max, N, N), K[i][j] is kernel value of depth i + 1 with first j layers fixed 
def kernel_value_batch(X, d_max): 
    K = np.zeros((d_max, d_max, X.shape[0], X.shape[0]))
    for fix_dep in range(d_max):
        S = np.matmul(X, X.T)
        H = np.zeros_like(S)
        for dep in range(d_max):
            if fix_dep <= dep:
                H += S
            K[dep][fix_dep] = H
            L = np.diag(S)
            P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
            Sn = np.clip(S / P, a_min = -1, a_max = 1)
            S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi
            H = H * (math.pi - np.arccos(Sn)) / 2.0 / math.pi
    return K

# return an array K of size (N, N), depth d_max, first fix_dep layers fixed
def kernel_value(X, d_max, fix_dep): 
    K = np.zeros((d_max, X.shape[0], X.shape[0]))
    S = np.matmul(X, X.T)
    H = np.zeros_like(S)
    for dep in range(d_max):
        if fix_dep <= dep:
            H += S
        K[dep] = H
        L = np.diag(S)
        P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = np.clip(S / P, a_min = -1, a_max = 1)
        S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi
        H = H * (math.pi - np.arccos(Sn)) / 2.0 / math.pi
    return K[d_max - 1]
