import numpy as np
from numba import njit
from numpy import linalg as la

import logging

@njit()
def get_coefficients(x, y):
    D1 = np.zeros((x.shape[0], 3))
    D1[:, 0] = x ** 2
    D1[:, 1] = x * y
    D1[:, 2] = y ** 2

    D2 = np.zeros_like(D1)
    D2[:, 0] = x
    D2[:, 1] = y
    D2[:, 2] = np.ones_like(x)

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])
    M = la.inv(C1) @ (S1 - S2 @ la.inv(S3) @ S2.T)

    eigval, eigvec = la.eig(M)

    cond = (
        4*np.multiply(eigvec[0, :], eigvec[2, :])
        - np.power(eigvec[1, :], 2)
    )
    a1 = eigvec[:, np.nonzero(cond > 0)[0]]

    a2 = la.inv(-S3) @ S2.T @ a1

    a1_len = a1.shape[0]
    a2_len = a2.shape[0]

    coef_ = np.zeros((a1_len + a2_len, 1))
    coef_[0:a1_len, 0] = a1[:, 0]
    coef_[a1_len:(a1_len + a2_len), 0] = a2[:, 0]

    return np.asarray(coef_).ravel()

def fit(X):
    # extract x-y pairs
    x, y = X.T

    coefficients = get_coefficients(x, y)
   

    # Eigenvectors are the coefficients of an ellipse in general form
    # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0
    # [eqn. 15) from (**) or (***)
    a = coefficients[0]
    b = coefficients[1] / 2.
    c = coefficients[2]
    d = coefficients[3] / 2.
    f = coefficients[4] / 2.
    g = coefficients[5]

    # Finding center of ellipse [eqn.19 and 20] from (**)
    x0 = (c*d - b*f) / (b**2 - a*c)
    y0 = (a*f - b*d) / (b**2 - a*c)
    center = [x0, y0]

    # Find the semi-axes lengths [eqn. 21 and 22] from (**)
    numerator = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    denominator1 = (b*b - a*c) * (
        (c-a) * np.sqrt(1+4*b**2 / ((a-c)*(a-c))) - (c+a)
    )
    denominator2 = (b*b - a*c) * (
        (a-c) * np.sqrt(1+4*b**2 / ((a-c) * (a-c))) - (c+a)
    )
    width = np.sqrt(numerator / denominator1)
    height = np.sqrt(numerator / denominator2)

    # Angle of counterclockwise rotation of major-axis of ellipse to x-axis
    # [eqn. 23] from (**) or [eqn. 26] from (***).
    phi = .5 * np.arctan((2.*b) / (a-c))

    return center, width, height, phi