import cupy as cp
import numpy as np
from cupy import linalg as la

def get_coefficients(x, y):
    x_on_gpu0 = cp.asarray(x, dtype=cp.float32)
    y_on_gpu0 = cp.asarray(y, dtype=cp.float32)

    x_m_on_gpu0 = cp.mean(x_on_gpu0, dtype=cp.float32)
    y_m_on_gpu0 = cp.mean(y_on_gpu0, dtype=cp.float32)

    u = cp.zeros_like(x_on_gpu0, dtype=cp.float32)
    v = cp.zeros_like(y_on_gpu0, dtype=cp.float32)

    u = x_on_gpu0 - x_m_on_gpu0
    v = y_on_gpu0 - y_m_on_gpu0

    Suv  = cp.sum(u * v, dtype=cp.float32)
    Suu  = cp.sum(u * u, dtype=cp.float32)
    Svv  = cp.sum(v * v, dtype=cp.float32)
    Suuv = cp.sum(u * u * v, dtype=cp.float32)
    Suvv = cp.sum(u * v * v, dtype=cp.float32)
    Suuu = cp.sum(u * u * u, dtype=cp.float32)
    Svvv = cp.sum(v * v * v, dtype=cp.float32)

    A = cp.array([ [Suu, Suv], [Suv, Svv]], dtype=cp.float32)
    b = cp.array([Suuu + Suvv, Svvv + Suuv], dtype=cp.float32) / 2
    uc, vc = cp.dot(la.inv(A), b)

    xc_1 = uc + x_m_on_gpu0
    yc_1 = vc + y_m_on_gpu0

    Ri_1 = cp.sqrt((x_on_gpu0 - xc_1)**2 + (y_on_gpu0 - yc_1)**2)
    R_1 = cp.mean(Ri_1) 
    return [xc_1.get(), yc_1.get(), R_1.get(), R_1.get(), 0]
    


def fit(X):
    x, y = X.T
    return get_coefficients(x, y)