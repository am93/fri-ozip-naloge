from time import time

import numpy as np
import matplotlib.pyplot as plt
from math import pow
import random
from matplotlib import colors as mcolors


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

            # prevent division by zero
            if r == 0.0:
                r = 1e-5

            if y[i] == y[j]:
                fij = -r
            else:
                fij = 1/r

            F[i, 0] += fij * (dx/r)
            F[j, 0] -= fij * (dx/r)

            F[i, 1] += fij * (dy/r)
            F[j, 1] -= fij * (dy/r)

    # compute gradient
    G = X.T.dot(F)
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
    # initialize random projection matrix A
    A = np.random.rand(X.shape[1],2) * 2 - 1

    iter = 0
    convergence = False

    # Loop until convergence
    while iter < maxiter and not convergence:
        P = X.dot(A)
        G = grad(X, y, P)
        # gradient normalization
        coeff = np.min(np.linalg.norm(A, axis=1) / np.linalg.norm(G, axis=1))
        step = 0.1 * coeff
        A_new = A + step * G
        iter += 1
        #plot(X,y,A)
        print('------------------------------------------------> sum(G): ', np.linalg.norm(A - A_new))
        A = A_new

    return A

def plot(X, Y, A, classname):

    P = X.dot(A)

    colors = ['aquamarine', 'yellowgreen', 'chartreuse', 'coral',
              'cadetblue', 'darkviolet', 'red', 'olive', 'orchid',
              'seagreen', 'navy', 'yellow', 'orange', 'maroon']

    fig, ax = plt.subplots()

    points = {}

    for y in np.unique(Y):
        points[y] = [[], []]
        for i in range(P.shape[0]):
            # check if it is current class
            if y == Y[i]:
                points[y][0].append(P[i, 0])
                points[y][1].append(P[i, 1])

    for y in np.unique(Y):
        col = random.choice(colors)
        colors.remove(col)
        ax.scatter(points[y][0], points[y][1], c=[mcolors.CSS4_COLORS[col]] * (len(points[y][0])), label=classname[y.astype(int)], alpha=0.65)

    ax.legend()
    ax.grid(False)
    plt.show()


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('zoo')
    #data = Orange.preprocess.Normalize()(data)
    X, y = data.X, data.Y
    classname = data.domain._variables[-1].values

    t = time()
    A = freeviz(X, y, maxiter=300)
    print('time', time() - t)
    plot(X,y,A, classname)
