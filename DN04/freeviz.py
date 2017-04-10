from time import time

from sklearn.metrics import silhouette_score
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
    A = np.random.rand(X.shape[1],2) * 20 - 10

    global attributes

    iter = 0
    convergence = False

    # Loop until convergence
    while iter < maxiter and not convergence:
        P = X.dot(A)
        G = grad(X, y, P)

        # gradient normalization
        coeff = np.min(np.linalg.norm(A, axis=1) / (np.linalg.norm(G, axis=1)+1e-7))
        A_new = A + 0.1 * coeff * G

        # scaling
        scale_fac = np.max(np.linalg.norm(A_new, axis=1))
        if scale_fac > 0:
            A_new /= scale_fac

        diff = np.linalg.norm(A - A_new)
        A = A_new

        # check convergence
        print('--> sum(G): ', diff)
        if diff < 0.003:
            print('Converged at iteration: ', iter)
            convergence = True
        else:
            iter += 1


    return A

def plot(X, Y, A, classnames=None, attributes=None, max_attr=5, title=None):
    """
    Function plots FreeViz data projections and visualizes base vectors
    :param X: data points [n_examples, n_features]
    :param Y: class values [n_examples, 1]
    :param A: projection matrix computed with FreeViz [n_features, 2]
    :param classnames: string values for class names
    :param attributes: string values for feature names (base vectors)
    :param max_attr: maximum number of base vectors to be visualized
    """

    if A is not None:
        # project points
        P = X.dot(A)

        # pick only largest base vectors
        vecs_idx = [x[0] for x in sorted(enumerate(np.linalg.norm(A, axis=1)), key = lambda x: x[1], reverse=True)][:max_attr]
    else:
        P = X

    # scaling
    scale_fac = np.max(np.linalg.norm(P, axis=1))
    if scale_fac > 0:
        P /= scale_fac

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

    # plot projections
    for y in np.unique(Y):
        col = random.choice(colors)
        colors.remove(col)
        # use string class names in labels
        if classnames is not None:
            ax.scatter(points[y][0], points[y][1], c=[mcolors.CSS4_COLORS[col]] * (len(points[y][0])),
                       label=classnames[y.astype(int)], alpha=0.65)
        else:
            ax.scatter(points[y][0], points[y][1], c=[mcolors.CSS4_COLORS[col]] * (len(points[y][0])),
                       label=y.astype(int), alpha=0.65)

    # display base vectors
    if attributes is not None:
        for i,a in enumerate(attributes):
            if i < max_attr:
                idx = vecs_idx[i]
                plt.plot([0, A[idx, 0]], [0, A[idx, 1]], 'k-')
                plt.text(A[idx, 0], A[idx, 1], a)

    ax.legend()
    ax.grid(False)
    ax.axis('off')
    if title is not None:
        plt.title('Projekcija podatkov {0} (score:{1:.4g})'.format(title[0],title[1]))
    plt.show()


def evaluate_projection(P, y):
    """
    [BONUS NALOGA]: Funkcija kvantitativno evaluira kvaliteto projekcije na podlagi ideje iz grucenja. Ker zelimo s
    pomocjo projekcije odkriti podobnosti oz. podobne lastnosti primerov iz istih razredov, bi radi, da se primeri pri
    projekciji zdruzijo v gruce. Torej, da so primeri istega razreda skupaj na majhni razdalji in hkrati oddaljeni od
    primerov drugih razredov. Pri grucenju uporabljamo mero silhueta, ki ima vrednosti na intervalu [-1, 1] in ocenjuje
    ravno to kar smo zapisali zgoraj. Vrednost 1 pomeni, da je primer dobro poziconiran znotraj svoje gruce in dalec od
    ostalih gruc, medtem ko vrednost -1 kaze ravno nasprotno.

    Rezultat funkcije je povprecna vrednost silhuete za podane projekcije.
    
    KOMENTAR REZULTATOV: 
    Ce primerjamo rezultate na podatkih MNIST-1K vidimo, da je FreeViz razbil podatke na mnogo bolj
    cloveku razmuljive gruce. Pri pristopu PCA se gruce mnogo bolj prekrivajo in niso dobro razvidne, zato je ocena (silhueta)
    tudi mnogo nižja in celo negativna - to pomeni, da je v povprecju primer blizje primerom iz druge gruce kot gruce 
    kateri pripada na podlagi razreda.
    Rezultati na podatkih ZOO so mnogo bolj podobni, podobni sta tudi vrednosti silhuet. Zopet je pristop z uporabo
    FreeViz dosegel visje rezultate.

    :param P: projected data points [n_examples, 2]
    :param y: class values [n_examples]
    :return: mean silhuette value
    """
    return silhouette_score(P, y)


if __name__ == '__main__':
    import Orange

    name = 'zoo'
    data = Orange.data.Table(name)
    data = Orange.preprocess.Normalize()(data)
    X, y = data.X, data.Y
    classnames = data.domain._variables[-1].values
    attributes = [a.name for a in data.domain._variables[:-1]]

    t = time()
    A = freeviz(X, y, maxiter=850)
    print('time', time() - t)
    evaluated = evaluate_projection(X.dot(A),y)
    plot(X,y,A, classnames, attributes, title=(name,evaluated))
    print('Projection score evaluation: ',evaluated)
