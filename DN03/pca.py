from time import time
import numpy as np
import Orange
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import silhouette_score
import random

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1.T, v2)
    sinang = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(cosang/sinang)


def pca_full(X):
    """
    Calculate full PCA transformation by using Numpy's np.linalg.eigh.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]

    Returns:
        eigenvectors (np.ndarray): Array of shape [n_features, n_features]
            containing eigenvectors as columns. The ith column should
            contain eigenvector corresponding to the ith largest eigenvalue.
    """
    x_avg = (np.sum(X, axis=0) / X.shape[0])[None, :]           # average example - column vector
    cov_matrix = np.zeros((X.shape[1], X.shape[1]))
    for row in X:
        row = row[None, :]                                      # make row also a valid column vector
        cov_matrix += (row.T - x_avg.T).dot((row + x_avg))
    cov_matrix /= X.shape[0]

    # compute eigenvectors from covariance matrix (return in descending order)
    return np.flip(np.linalg.eigh(cov_matrix)[1], axis=1)


def gram_schmidt_orthogonalize(vecs):
    """
    Gram-Schmidt orthogonalization of column vectors.

    Args:
        vecs (np.adarray): Array of shape [n_features, k] with column
            vectors to orthogonalize.

    Returns:
        Orthogonalized vectors of the same shape as on input.
    """
    # prepare first vector q in advance
    Q = (vecs[:, 0] / np.linalg.norm(vecs[:, 0]))[:, None]
    for j in range(1, vecs.shape[1]):
        vj = vecs[:, j]
        for i in range(0, j):
            rij = (Q[:, i].T.dot(vj))
            vj -= rij * Q[:, i]

        Q = np.column_stack((Q, vj / np.linalg.norm(vj)))
    return Q


def pca_2d(X, eps=1e-5):
    """
    Calculate the first two components of PCA transformation by using
    the power method with Gram-Schmidt orthogonalization.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]
        eps (float): Stopping criterion threshold for Frobenius norm.

    Returns:
        eigenvectors (np.ndarray): Array of shape [n_features, 2]
            containing the eigenvectors corresponding to the largest and
            the second largest eigenvalues.
    """
    # eigenvector initialization
    eivec = np.random.rand(X.shape[1], 2) - 0.5
    eivec = gram_schmidt_orthogonalize(eivec / np.linalg.norm(eivec, axis=0))
    eivec_old = eivec
    diff_old = 10000

    # covariance matrix
    M = np.cov(X.T)

    # repeat until convergence
    while(True):
        eivec = M.dot(eivec)
        eivec = gram_schmidt_orthogonalize(eivec / np.linalg.norm(eivec, axis=0)) # normalize and ortogonalize

        # check for convergence
        diff = np.linalg.norm(eivec - eivec_old)
        if np.abs(diff) < eps:
            break
        else:
            diff_old = diff
            eivec_old = eivec

    # compute eigenvalues
    lambda1 = np.sum(eivec[:,:1].T.dot(M).dot(eivec[:,:1]))     # np.sum is here just as hack to get scalar
    lambda2 = np.sum(eivec[:,1:2].T.dot(M).dot(eivec[:,1:2]))   # np.sum is here just as hack to get scalar

    if lambda1 > lambda2:
        return eivec
    else:
        return np.flip(eivec, axis=1)


def project_data(X, vecs):
    """
    Project the data points in X into the subspace of eigenvectors.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]
        vecs: Array of shape [n_features, k] containing the eigenvectors.

    Returns:
        np.ndarray: Projected data of shape [n_examples, k].
    """
    return X.dot(vecs)


def evaluate_projection(P, y):
    """
    [BONUS NALOGA]: Funkcija kvantitativno evaluira kvaliteto projekcije na podlagi ideje iz grucenja. Ker zelimo s
    pomocjo projekcije odkriti podobnosti oz. podobne lastnosti primerov iz istih razredov, bi radi, da se primeri pri
    projekciji zdruzijo v gruce. Torej, da so primeri istega razreda skupaj na majhni razdalji in hkrati oddaljeni od
    primerov drugih razredov. Pri grucenju uporabljamo mero silhueta, ki ima vrednosti na intervalu [-1, 1] in ocenjuje
    ravno to kar smo zapisali zgoraj. Vrednost 1 pomeni, da je primer dobro poziconiran znotraj svoje gruce in dalec od
    ostalih gruc, medtem ko vrednost -1 kaze ravno nasprotno.

    Rezultat funkcije je povprecna vrednost silhuete za podane projekcije.

    :param P: projected data points [n_examples, 2]
    :param y: class values [n_examples]
    :return: mean silhuette value
    """
    return silhouette_score(P, y)


def visualize_data(X, Y, filename='default.pdf', title=None, classnames=None):

    colors = ['aquamarine', 'yellowgreen', 'chartreuse', 'coral',
              'cadetblue', 'darkviolet', 'red', 'olive', 'orchid',
              'seagreen', 'navy', 'yellow', 'orange','maroon']
    fig, ax = plt.subplots()

    points = {}

    for y in np.unique(Y):
        points[y] = [[],[]]
        for i in range(X.shape[0]):
            # check if it is current class
            if y == Y[i]:
                points[y][0].append(X[i,0])
                points[y][1].append(X[i,1])

    for y in np.unique(Y):
        col = random.choice(colors)
        colors.remove(col)
        if classnames is not None:
            ax.scatter(points[y][0],points[y][1], c=[mcolors.CSS4_COLORS[col]]*(len(points[y][0])), label=classnames[y.astype(int)], alpha=0.65)
        else:
            ax.scatter(points[y][0], points[y][1], c=[mcolors.CSS4_COLORS[col]] * (len(points[y][0])), label=y,alpha=0.65)

    ax.legend()
    ax.grid(True)
    if title is not None:
        plt.title('PCA projekcija podatkov {0} (score:{1:.4g})'.format(title[0],title[1]))
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    data = Orange.data.Table('mnist-1k.tab')

    t1 = time()
    vecs_np = pca_full(data.X)
    print('Full time: {:.4f}s'.format(time() - t1))
    transformed_numpy = project_data(data.X, vecs_np[:, :2])
    visualize_data(transformed_numpy, data.Y, 'pca_full.pdf')

    t1 = time()
    vecs_pow = pca_2d(data.X)
    print('2D time: {:.4f}s'.format(time() - t1))
    transformed_power = project_data(data.X, vecs_pow)
    score = evaluate_projection(transformed_power, data.Y)
    classnames = data.domain._variables[-1].values
    visualize_data(transformed_power, data.Y, 'pca_2d.pdf', title=('MNIST-1K',score), classnames=classnames)
