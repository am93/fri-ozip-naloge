from time import time
import numpy as np
import Orange


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

    # compute eigenvectors from covariance matrix
    return  np.linalg.eigh(cov_matrix)


def gram_schmidt_orthogonalize(vecs):
    """
    Gram-Schmidt orthogonalization of column vectors.

    Args:
        vecs (np.adarray): Array of shape [n_features, k] with column
            vectors to orthogonalize.

    Returns:
        Orthogonalized vectors of the same shape as on input.
    """
    pass


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
    pass


def project_data(X, vecs):
    """
    Project the data points in X into the subspace of eigenvectors.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]
        vecs: Array of shape [n_features, k] containing the eigenvectors.

    Returns:
        np.ndarray: Projected data of shape [n_examples, k].
    """
    pass


if __name__ == '__main__':
    data = Orange.data.Table('mnist-1k.tab')

    t1 = time()
    vecs_np = pca_full(data.X)
    print('Full time: {:.4f}s'.format(time() - t1))
    transformed_numpy = project_data(data.X, vecs_np[:, :2])

    t1 = time()
    vecs_pow = pca_2d(data.X)
    print('2D time: {:.4f}s'.format(time() - t1))
    transformed_power = project_data(data.X, vecs_pow)
