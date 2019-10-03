import math
import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score

def ridge_regression(K1, K2, y1, y2, alpha, c):
    n_val, n_train = K2.shape
    clf = KernelRidge(kernel = "precomputed", alpha = alpha)
    one_hot_label = np.eye(c)[y1] - 1.0 / c
    clf.fit(K1, one_hot_label)
    z = clf.predict(K2).argmax(axis = 1)
    return 1.0 * np.sum(z == y2) / n_val

def svm(K1, K2, y1, y2, C, c):
    n_val, n_train = K2.shape
    clf = SVC(kernel = "precomputed", C = C, cache_size = 100000)
    clf.fit(K1, y1)
    z = clf.predict(K2)
    return 1.0 * np.sum(z == y2) / n_val

def gen_bound(K, y):
    alpha = np.linalg.solve(K, y)
    C = alpha.T.dot(K).dot(alpha)
    return np.sum(np.sqrt(np.diag(C))) * np.sqrt(np.trace(K)) / K.shape[0]

def normalize(K):
    L = np.diag(K)
    return K / np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)

def translate(K1, K2):
    n1, n2 = K2.shape
    m1 = np.mean(K1, axis = 0)
    m2 = np.mean(K2, axis = 0)
    o1 = np.ones(n1)
    o2 = np.ones(n2)
    return K1 - np.outer(m1, o1) - np.outer(o1, m1) + np.mean(K1), K2 - np.outer(m1, o2) - np.outer(o1, m2) + np.mean(K1)

