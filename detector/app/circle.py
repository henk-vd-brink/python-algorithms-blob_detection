import os
import numpy as np
# import numba, os
from numpy import linalg as la

import logging, time

# numba.config.DISABLE_JIT = True if os.environ.get("DISABLE_JIT") in ["True", "true", 1] else False

# @numba.njit()
def get_coefficients(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    u = np.zeros_like(x)
    v = np.zeros_like(y)

    u = x - x_m
    v = y - y_m

    Suv  = np.sum(u * v)
    Suu  = np.sum(u * u)
    Svv  = np.sum(v * v)
    Suuv = np.sum(u * u * v)
    Suvv = np.sum(u * v * v)
    Suuu = np.sum(u * u * u)
    Svvv = np.sum(v * v * v)

    A = np.array([ [Suu, Suv], [Suv, Svv]])
    b = np.array([Suuu + Suvv, Svvv + Suuv]) / 2
    uc, vc = np.dot(la.inv(A), b)

    xc_1 = uc + x_m
    yc_1 = vc + y_m

    Ri_1 = np.sqrt((x - xc_1)**2 + (y - yc_1)**2)
    R_1 = np.mean(Ri_1)    
    return np.array([xc_1, yc_1, R_1, R_1, 0])

def fit(X):
    x, y = X.T
    return get_coefficients(x, y)