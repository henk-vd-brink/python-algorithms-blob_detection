import numpy as np

def fit(X):
    # extract x-y pairs
    x, y = X.T

    # Quadratic part of design matrix [eqn. 15] from (*)
    D1 = np.vstack([x**2, x * y, y**2]).T
    # Linear part of design matrix [eqn. 16] from (*)
    D2 = np.vstack([x, y, np.ones_like(x)]).T

    # Forming scatter matrix [eqn. 17] from (*)
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Constraint matrix [eqn. 18]
    C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

    # Reduced scatter matrix [eqn. 29]
    M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)

    # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this
    # equation [eqn. 28]
    eigval, eigvec = np.linalg.eig(M)

    # Eigenvector must meet constraint 4ac - b^2 to be valid.
    cond = (
        4*np.multiply(eigvec[0, :], eigvec[2, :])
        - np.power(eigvec[1, :], 2)
    )
    a1 = eigvec[:, np.nonzero(cond > 0)[0]]

    # |d f g> = -S3^(-1) * S2^(T)*|a b c> [eqn. 24]
    a2 = np.linalg.inv(-S3) @ S2.T @ a1

    # Eigenvectors |a b c d f g>
    # list of the coefficients describing an ellipse [a,b,c,d,f,g]
    # corresponding to ax**2 + 2bxy + cy**2 + 2dx + 2fy + g
    coef_ = np.vstack([a1, a2])
    coefficients = np.asarray(coef_).ravel()

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