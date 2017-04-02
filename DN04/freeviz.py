from time import time

import numpy as np
import matplotlib.pyplot as plt
from math import pow


def grad(X, y, P):
    """
    Calculate FreeViz gradients.

    Args:
        X (np.ndarray): data matrix of shape [n_examples, n_features]
        y (np.ndarray): target variable of shape [n_examples]
        P (np.ndarray): matrix with projected data of shape [n_examples, 2]

    Temp:
        F (np.ndarray): forces matrix

    Returns:
        np.ndarray: gradients shape [n_features, 2]
    """
    # initialize forces matrix
    F = np.zeros(P.shape)

    # compute all forces between points
    for i in range(0, P.shape[0]):
        for j in range(i+1, P.shape[0]):
            dx = P[i, 0] - P[j, 0]
            dy = P[i, 1] - P[j, 1]
            r = np.sqrt(pow(dx,2) + pow(dy,2))

            if y[i] == y[j]:
                fij = -r
            else:
                fij = 1/r

            F[i, 0] += fij * (dx/r)
            F[j, 0] -= fij * (dx/r)

            F[i, 1] += fij * (dy/r)
            F[i, 1] -= fij * (dy/r)

    # compute gradient
    G = -1 * X.t.dot(F)
    return G



def freeviz(X, y, maxiter=100):
    """
    Find FreeViz projection.

    Args:
        X (np.ndarray): data matrix of shape [n_examples, n_features]
        y (np.ndarray): target variable of shape [n_examples]
        maxiter (int): maximum number of iterations

    Returns:
        np.ndarray: projection matrix A of shape [n_features, 2]
    """
    pass


def plot( add_required_parameters ):
    pass


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('mnist-1k')
    data = Orange.preprocess.Normalize()(data)
    X, y = data.X, data.Y

    t = time()
    A = freeviz(X, y, maxiter=300)
    print('time', time() - t)
    plot(...)
