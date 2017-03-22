import numpy as np
from scipy.linalg import hilbert


#def gramschmidt(A):
#    """ Gram-Schmidt orthogonalization of column-vectors. Matrix A passes
#    vectors in its columns, orthonormal system is returned in columns of
#    matrix Q. """
#    _, k = A.shape

    # first basis vector
#    Q = A[:, [0]] / np.linalg.norm(A[:, 0])
#    for j in range(1, k):
#        # orthogonal projection, loop-free implementation
#        q = A[:, j] - np.dot(Q, np.dot(Q.T, A[:, j]))

        # check premature termination
#        nq = np.linalg.norm(q)
        # add new basis vector as another column of Q
#        Q = np.column_stack([Q, q / nq])
#    return Q


def gramschmidt(vecs):
    """
    Gram-Schmidt orthogonalization of column vectors.

    Args:
        vecs (np.adarray): Array of shape [n_features, k] with column
            vectors to orthogonalize.

    Returns:
        Orthogonalized vectors of the same shape as on input.
    """
    # prepare first vector q in advance
    while(True):
        Q = (vecs[:, 0] / np.linalg.norm(vecs[:, 0]))[:, None]
        for j in range(1, vecs.shape[1]):
            vj = vecs[:, j]
            for i in range(0, j):
                rij = (Q[:, i].T.dot(vj))
                vj -= rij * Q[:, i]

            Q = np.column_stack((vj / np.linalg.norm(vj), Q))

        return Q

    return Q

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def stabGramSchmidt(X):
    rowsA, colsA = X.shape
    # Initialize the Y vector
    Y = np.zeros(rowsA)
    # Initialize Q, the array of vectors which are the vectors orthogonal to the vectors in X
    Q = np.zeros([rowsA, colsA])
    # Step 1
    # np.inner is the inner product of vectors for 1-D arrays
    # X[:,0] gives us the first column vector
    productX1 = 1.0 / np.sqrt(np.inner(X[:, 0], X[:, 0]))
    # Steps 2 through 5
    Q[:, 0] = productX1 * X[:, 0]
    for j in range(1, colsA):
        # Step 3
        Y = X[:, j]
        for i in range(0, j):
            Y = Y - np.inner(X[:, j], Q[:, i]) * Q[:, i]
        # Step 4
        productYj = 1.0 / np.sqrt(np.inner(Y, Y))
        Q[:, j] = productYj * Y
    return Q


def main():
    """ Main function, demonstrates roundoff on the result of the Gram-Schmidt
    orthogonalization. """
    # set print options to use lower precision
    printopt = np.get_printoptions()
    np.set_printoptions(formatter={'float': '{:8.2g}'.format}, linewidth=200)

    # create special matrix, the so-called Hilbert-matrix Aij = 1 / (i + j + 1)
    A = np.random.rand(10,2) - 0.5
    print(A)
    Q = gramschmidt(A)
    print(Q)

    # matrix according to theory should be unit matrix:
    I = np.dot(Q.T, Q)
    print('I = \n{}'.format(I))

    # numpy's internal orthogonaliztation by QR-decomposition
    Q1, R1 = np.linalg.qr(A)
    D = A - np.dot(Q1, R1)
    print('D = \n{}'.format(D))
    I1 = np.dot(Q1.T, Q1)
    print('I1 = \n{}'.format(I1))

    np.set_printoptions(**printopt)


if __name__ == '__main__':
    main()
